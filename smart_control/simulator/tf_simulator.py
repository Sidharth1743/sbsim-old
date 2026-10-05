"""Tensorflow-enabled Finite Difference calculator.

Iterative methods that loop through each control volume sequentially
are very slow with geometries that have many control volumes (CV). The
TFSimulator, instead, processes the finite differences as a set of
tensor operations.
"""

import enum
import functools
from typing import Mapping, Optional, Sequence

from absl import logging
import gin
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.eager import context as tf_context  # pylint: disable=g-direct-tensorflow-import

from smart_control.simulator import building as building_py
from smart_control.simulator import cuda_fd
from smart_control.simulator import hvac_floorplan_based as hvac_py
from smart_control.simulator import simulator_flexible_floor_plan as simulator
from smart_control.simulator import weather_controller as weather_controller_py

# List of neighbors for a specific CV.
NeighborCoordinates = Sequence[simulator.CVCoordinates]
# Neighbors, indexable by a specific i,j coord.
Neighbors = Sequence[Sequence[NeighborCoordinates]]


# All CVs must be interior, exterior, or boundary positions.
class CVPositionType(enum.Enum):
  INTERIOR = 1
  EXTERIOR = 2
  BOUNDARY = 3


# All Boundary CVs must be either edge (three non-exterior neighbors), or
# corner (two non-exterior neighbors).
class CVBoundaryType(enum.Enum):
  EDGE = 1
  CORNER = 2


# All edge CVs must be top, bottom, left or right edges.
class CVEdgeOrientationType(enum.Enum):
  """Classifies edge CVs by orientation."""

  TOP = (1, 0.5, 1.0)
  BOTTOM = (2, 0.5, 1.0)
  LEFT = (3, 1.0, 0.5)
  RIGHT = (4, 1.0, 0.5)

  def __init__(
      self,
      value: int,
      vertical_scale_factor: float,
      horizontal_scale_factor: float,
  ):
    self._value_ = value
    self.vertical_scale_factor = vertical_scale_factor
    self.horizontal_scale_factor = horizontal_scale_factor


# All corner CVs must be top left, bottom right, bottom left, or top right.
class CVCornerOrientationType(enum.Enum):
  TOP_LEFT = (1, 0.5)
  BOTTOM_LEFT = (2, 0.5)
  BOTTOM_RIGHT = (3, 0.5)
  TOP_RIGHT = (4, 0.5)

  def __init__(self, value: int, scale_factor: float):
    self._value_ = value
    self.scale_factor = scale_factor


class CVType:
  """Classifies a CV based on its neighborhood and orientation.

  A CV can be interior with four neighbors, exterior with zero or one
  neighbor, or boundary with two or three neighbors. A boundary CV
  can be an edge with three neighbors, or a corner with two neighbors.

  Using a regular class instead of a data class to ensure consistent
  assignment of CV proprties.
  """

  def __init__(
      self,
      position: CVPositionType,
      boundary: Optional[CVBoundaryType] = None,
      edge: Optional[CVEdgeOrientationType] = None,
      corner: Optional[CVCornerOrientationType] = None,
  ):
    self.position = position
    self.boundary = boundary
    self.edge = edge
    self.corner = corner

    if self.position in [CVPositionType.EXTERIOR, CVPositionType.INTERIOR]:
      if (
          self.boundary is not None
          or self.edge is not None
          or self.corner is not None
      ):
        raise ValueError(
            'Cannot assign boundary properties to a non-boundary CV.'
        )
    elif self.position == CVPositionType.BOUNDARY:
      if self.edge is None and self.corner is None:
        raise ValueError('Boundary CVs must have edge or corner properties')

      if self.edge is not None and self.corner is not None:
        raise ValueError('Boundary CVs cannot be edge AND corner.')

      if self.boundary == CVBoundaryType.EDGE and self.edge is None:
        raise ValueError('Edge CVs must be assigned an edge orientation.')

      if self.boundary == CVBoundaryType.CORNER and self.corner is None:
        raise ValueError('Corner CVs must be assigned a corner orientation.')

  def __eq__(self, other) -> bool:

    if isinstance(other, CVType):
      return (
          self.position == other.position
          and self.boundary == other.boundary
          and self.edge == other.edge
          and self.corner == other.corner
      )
    return False

  @property
  def horizontal_scale_factor(self) -> float:
    """Returns the horizontal scale factor for this CV."""
    if self.position != CVPositionType.BOUNDARY:
      return 1.0
    match self.boundary:
      case CVBoundaryType.CORNER:
        if self.corner is None:
          raise ValueError('Corner CVs must be assigned a corner orientation.')
        return self.corner.scale_factor
      case CVBoundaryType.EDGE:
        if self.edge is None:
          raise ValueError('Edge CVs must be assigned an edge orientation.')
        return self.edge.horizontal_scale_factor
    raise ValueError('horizontal_scale_factor is not defined for this CV type.')

  @property
  def vertical_scale_factor(self) -> float:
    """Returns the vertical scale factor for this CV."""
    if self.position != CVPositionType.BOUNDARY:
      return 1.0
    match self.boundary:
      case CVBoundaryType.CORNER:
        if self.corner is None:
          raise ValueError('Corner CVs must be assigned a corner orientation.')
        return self.corner.scale_factor
      case CVBoundaryType.EDGE:
        if self.edge is None:
          raise ValueError('Edge CVs must be assigned an edge orientation.')
        return self.edge.vertical_scale_factor
    raise ValueError('vertical_scale_factor is not defined for this CV type.')


BoundaryCVMapping = Mapping[simulator.CVCoordinates, CVType]


def classify_cv(
    coords: simulator.CVCoordinates, neighbors: Neighbors
) -> CVType:
  """Classifies a CV based on its neighborhood.

  0-1 neighbors: EXTERIOR
  2 neighbors: CORNER
  3 neighbors: EDGE
  4 neighbors: INTERIOR

  Args:
    coords: i, j coords of the CV
    neighbors:  a lookup table of all the neighbors.

  Returns:
    a string that represents the CV
  """

  def _cv_type_corner_factory(
      corner: CVCornerOrientationType,
  ) -> CVType:

    return CVType(
        position=CVPositionType.BOUNDARY,
        boundary=CVBoundaryType.CORNER,
        corner=corner,
    )

  def _cv_corner_type(cv_neighbors: Sequence[tuple[int, int]]) -> CVType:
    if set([(i + 1, j), (i, j + 1)]) == set(cv_neighbors):
      return _cv_type_corner_factory(corner=CVCornerOrientationType.TOP_LEFT)
    if set([(i - 1, j), (i, j + 1)]) == set(cv_neighbors):
      return _cv_type_corner_factory(corner=CVCornerOrientationType.BOTTOM_LEFT)
    if set([(i + 1, j), (i, j - 1)]) == set(cv_neighbors):
      return _cv_type_corner_factory(corner=CVCornerOrientationType.TOP_RIGHT)
    if set([(i - 1, j), (i, j - 1)]) == set(cv_neighbors):
      return _cv_type_corner_factory(
          corner=CVCornerOrientationType.BOTTOM_RIGHT
      )
    raise ValueError(
        f"Wasn't able to determine which corner the CV {(i, j)} is."
    )

  def _cv_type_edge_factory(
      edge: CVEdgeOrientationType,
  ) -> CVType:

    return CVType(
        position=CVPositionType.BOUNDARY,
        boundary=CVBoundaryType.EDGE,
        edge=edge,
    )

  def _cv_edge_type(cv_neighbors: Sequence[tuple[int, int]]) -> CVType:
    edge = functools.partial(_cv_type_edge_factory)
    if set([(i, j + 1), (i, j - 1), (i + 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.TOP)
    if set([(i, j - 1), (i, j + 1), (i - 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.BOTTOM)
    if set([(i - 1, j), (i, j + 1), (i + 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.LEFT)
    if set([(i - 1, j), (i, j - 1), (i + 1, j)]) == set(cv_neighbors):
      return edge(CVEdgeOrientationType.RIGHT)
    raise ValueError(f"Wasn't able to determine which edge the CV {(i, j)} is.")

  i, j = coords

  cv_neighbors = neighbors[i][j]
  match len(cv_neighbors):
    case 0:
      return CVType(position=CVPositionType.EXTERIOR)
    case 1:
      return CVType(position=CVPositionType.EXTERIOR)
    case 2:
      return _cv_corner_type(cv_neighbors)
    case 3:
      return _cv_edge_type(cv_neighbors)
    case 4:
      return CVType(position=CVPositionType.INTERIOR)
    case _:
      raise ValueError(
          f"Wasn't able to determine which CV type the CV {(i, j)} is."
      )


def get_cv_mapping(
    neighbors: Neighbors,
    position_criterion: CVPositionType = CVPositionType.BOUNDARY,
) -> BoundaryCVMapping:
  """Gets a map of all the non-interior cv_types."""

  boundary_cv_mapping = {}
  ni = len(neighbors)
  nj = len(neighbors[0])
  for i in range(ni):
    for j in range(nj):
      cv_type = classify_cv((i, j), neighbors)
      if cv_type.position == position_criterion:
        boundary_cv_mapping[(i, j)] = cv_type

  return boundary_cv_mapping


def get_cv_dimension_tensors(
    control_volume_cm: float,
    boundary_cv_mapping: BoundaryCVMapping,
    shape: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
  """Returns horizontal and vertical CV dimension tensors.

  Boundary control volumes have varying heights and widths, since they share
  a part with the outside. For example, edge boundary CVs have one side
  that is half width, depending on orientation, and corner CVs have both edges
  half width.

  Args:
    control_volume_cm: centimeters width of each interior control volume.
    boundary_cv_mapping: dict of all boundary elements.
    shape: shape of the target matrix.

  Returns:
    horizontal (u), vertical (v) dimension tensors.
  """

  def _compute_cv_dimension(cv_type: CVType) -> tuple[float, float]:
    """Sets the horizontal and vertical CV dimension for one CV."""

    return (
        control_volume_cm * cv_type.horizontal_scale_factor,
        control_volume_cm * cv_type.vertical_scale_factor,
    )

  horizontal_cv_dimension = np.full(shape, control_volume_cm, dtype=np.float32)
  vertical_cv_dimension = np.full(shape, control_volume_cm, dtype=np.float32)

  for (i, j), cv_type in boundary_cv_mapping.items():

    cv_horizontal_cv_dimension, cv_vertical_cv_dimension = (
        _compute_cv_dimension(cv_type)
    )
    vertical_cv_dimension[i][j] = cv_vertical_cv_dimension
    horizontal_cv_dimension[i][j] = cv_horizontal_cv_dimension

  t_horizontal_cv_dimension = tf.convert_to_tensor(
      horizontal_cv_dimension, dtype=tf.float32
  )
  t_vertical_cv_dimension = tf.convert_to_tensor(
      vertical_cv_dimension, dtype=tf.float32
  )
  return t_horizontal_cv_dimension, t_vertical_cv_dimension


def get_oriented_convection_coefficient_tensors(
    convection_coefficient_air: float,
    shape: tuple[int, int],
    boundary_cv_mapping: BoundaryCVMapping,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Returns oriented convection coefficient tensors.

  Forced convection (wind) is the primary means of exchanging heat
  across the boundary CVs.

  To accommodate tensor operations, we have to create edge-oriented tensors
  for convection.

  Args:
    convection_coefficient_air: Outside air convection coefficient.
    shape: Tensor shape - should be identical to the other tensors, i.e., temp
    boundary_cv_mapping: dict of boundary CVs.

  Returns:
    tensors for convection coeffs: left, right, top, bottom, all of shape
  """

  def _set_edge_convection_coefficient(cv_type: CVType) -> None:
    """Sets the convection coefficient for one CV."""
    match cv_type.edge:
      case CVEdgeOrientationType.TOP:
        h_top_edge[i][j] = convection_coefficient_air
      case CVEdgeOrientationType.LEFT:
        h_left_edge[i][j] = convection_coefficient_air
      case CVEdgeOrientationType.RIGHT:
        h_right_edge[i][j] = convection_coefficient_air
      case CVEdgeOrientationType.BOTTOM:
        h_bottom_edge[i][j] = convection_coefficient_air

  def _set_corner_convection_coefficient(cv_type: CVType) -> None:
    match cv_type.corner:
      case CVCornerOrientationType.TOP_LEFT:
        h_top_edge[i][j] = convection_coefficient_air
        h_left_edge[i][j] = convection_coefficient_air
      case CVCornerOrientationType.TOP_RIGHT:
        h_top_edge[i][j] = convection_coefficient_air
        h_right_edge[i][j] = convection_coefficient_air
      case CVCornerOrientationType.BOTTOM_LEFT:
        h_bottom_edge[i][j] = convection_coefficient_air
        h_left_edge[i][j] = convection_coefficient_air
      case CVCornerOrientationType.BOTTOM_RIGHT:
        h_bottom_edge[i][j] = convection_coefficient_air
        h_right_edge[i][j] = convection_coefficient_air

  h_left_edge = np.full(shape, 0, dtype=np.float32)
  h_right_edge = np.full(shape, 0, dtype=np.float32)
  h_top_edge = np.full(shape, 0, dtype=np.float32)
  h_bottom_edge = np.full(shape, 0, dtype=np.float32)

  for (i, j), cv_type in boundary_cv_mapping.items():
    if cv_type.position == CVPositionType.BOUNDARY:
      if cv_type.boundary == CVBoundaryType.EDGE:
        _set_edge_convection_coefficient(cv_type)

      if cv_type.boundary == CVBoundaryType.CORNER:
        _set_corner_convection_coefficient(cv_type)

  t_h_left_edge = tf.convert_to_tensor(h_left_edge, dtype=tf.float32)
  t_h_right_edge = tf.convert_to_tensor(h_right_edge, dtype=tf.float32)
  t_h_top_edge = tf.convert_to_tensor(h_top_edge, dtype=tf.float32)
  t_h_bottom_edge = tf.convert_to_tensor(h_bottom_edge, dtype=tf.float32)
  return t_h_left_edge, t_h_right_edge, t_h_top_edge, t_h_bottom_edge


def get_oriented_conductivity_tensors(
    conductivity: np.ndarray,
    boundary_cv_mapping: BoundaryCVMapping,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Returns oriented conductivity tensors.

  Conduction is a function of two adjacent CVs. On boundary CVs, there will be
  no conduction transfer, k = 0. On internal CVs there will be heat transfer,
  k > 0.

  The conductivity is split by sides of the CV, top, bottom, left and right.

  Args:
    conductivity: conductivity of the CV itself.
    boundary_cv_mapping: mapping that contains the coordinates of boundary CVs

  Returns:
    tensors for left, right, top, bottom conductivities.
  """

  k_left_edge = conductivity.copy()
  k_right_edge = conductivity.copy()
  k_top_edge = conductivity.copy()
  k_bottom_edge = conductivity.copy()

  for (i, j), cv_type in boundary_cv_mapping.items():
    if cv_type.position == CVPositionType.BOUNDARY:
      if cv_type.boundary == CVBoundaryType.EDGE:
        if cv_type.edge == CVEdgeOrientationType.TOP:
          k_top_edge[i][j] = 0.0
        if cv_type.edge == CVEdgeOrientationType.LEFT:
          k_left_edge[i][j] = 0.0
        if cv_type.edge == CVEdgeOrientationType.RIGHT:
          k_right_edge[i][j] = 0.0
        if cv_type.edge == CVEdgeOrientationType.BOTTOM:
          k_bottom_edge[i][j] = 0.0

      if cv_type.boundary == CVBoundaryType.CORNER:
        if cv_type.corner == CVCornerOrientationType.TOP_LEFT:
          k_top_edge[i][j] = 0.0
          k_left_edge[i][j] = 0.0
        if cv_type.corner == CVCornerOrientationType.TOP_RIGHT:
          k_top_edge[i][j] = 0.0
          k_right_edge[i][j] = 0.0
        if cv_type.corner == CVCornerOrientationType.BOTTOM_LEFT:
          k_bottom_edge[i][j] = 0.0
          k_left_edge[i][j] = 0.0
        if cv_type.corner == CVCornerOrientationType.BOTTOM_RIGHT:
          k_bottom_edge[i][j] = 0.0
          k_right_edge[i][j] = 0.0

  t_k_left_edge = tf.convert_to_tensor(k_left_edge, dtype=tf.float32)
  t_k_right_edge = tf.convert_to_tensor(k_right_edge, dtype=tf.float32)
  t_k_top_edge = tf.convert_to_tensor(k_top_edge, dtype=tf.float32)
  t_k_bottom_edge = tf.convert_to_tensor(k_bottom_edge, dtype=tf.float32)
  return t_k_left_edge, t_k_right_edge, t_k_top_edge, t_k_bottom_edge


def shift_tensor_right(
    x: tf.Tensor,
    padding_value: float = 0,
) -> tf.Tensor:
  t_right = tf.pad(x, [[0, 0], [1, 0]], constant_values=padding_value)
  return tf.boolean_mask(t_right, [True] * x.shape[1] + [False], axis=1)


def shift_tensor_left(
    x: tf.Tensor,
    padding_value: float = 0,
) -> tf.Tensor:
  t_left = tf.pad(x, [[0, 0], [0, 1]], constant_values=padding_value)
  return tf.boolean_mask(t_left, [False] + [True] * x.shape[1], axis=1)


def shift_tensor_up(
    x: tf.Tensor,
    padding_value: float = 0,
) -> tf.Tensor:
  t_up = tf.pad(x, [[0, 1], [0, 0]], constant_values=padding_value)
  return tf.boolean_mask(t_up, [False] + [True] * x.shape[0], axis=0)


def shift_tensor_down(
    x: tf.Tensor,
    padding_value: float = 0,
) -> tf.Tensor:
  t_down = tf.pad(x, [[1, 0], [0, 0]], constant_values=padding_value)
  return tf.boolean_mask(t_down, [True] * x.shape[0] + [False], axis=0)


def apply_exterior_temps(
    t_temps: tf.Tensor, temp_inf: float, t_exterior_temps_mask: tf.Tensor
) -> tf.Tensor:
  """Applies the temp_inf to all CVs that are masked as exterior."""
  t_exterior_temp = tf.fill(
      t_temps.shape, tf.constant(temp_inf, dtype=tf.float32)
  )
  return tf.where(t_exterior_temps_mask, t_exterior_temp, t_temps)


@gin.configurable
class TFSimulator(simulator.SimulatorFlexibleGeometries):
  """Tensor-based simulator that used matrix ops to update temps."""

  def __init__(
      self,
      building: building_py.Building,
      hvac: hvac_py.FloorPlanBasedHvac,
      weather_controller: weather_controller_py.WeatherController,
      time_step_sec: float,
      convergence_threshold: float,
      iteration_limit: int,
      iteration_warning: int,
      start_timestamp: pd.Timestamp,
      relative_convergence_threshold: float | None = 1e-6,
      relative_convergence_streak: int = 20,
      use_graph_fd_loop: bool = True,
      use_cuda_fd_kernel: bool = True,
      cuda_fd_fuse_k: int = 8,
      record_video: bool = True,
  ):

    super().__init__(
        building,
        hvac,
        weather_controller,
        time_step_sec,
        convergence_threshold,
        iteration_limit,
        iteration_warning,
        start_timestamp,
        relative_convergence_threshold=relative_convergence_threshold,
        relative_convergence_streak=relative_convergence_streak,
        record_video=record_video,
    )

    # Get a mapping of all the boundary CVs that interface between interior
    # and extrior CVs. This mapping will be used to process boundary elements
    # iteratively.
    self._boundary_cv_mapping = get_cv_mapping(
        building.neighbors, position_criterion=CVPositionType.BOUNDARY
    )
    n_boundary_elements = len(self._boundary_cv_mapping)
    logging.info('Number of boundary CVs: %d', n_boundary_elements)
    # Get a binary mask that mark exterior CVs so that they will always be
    # assigned ambinent air temps.
    self._t_exerior_temps_mask = self._get_tensor_exterior_mask(building)
    n_exterior_elements = tf.math.count_nonzero(self._t_exerior_temps_mask)
    logging.info('Number of exterior CVs: %d', n_exterior_elements)

    n_elements = self.building.temp.shape[0] * self.building.temp.shape[1]
    n_interior_elements = n_elements - n_boundary_elements - n_exterior_elements
    logging.info('Number of interior CVs: %d', n_interior_elements)

    self._t_u, self._t_v = get_cv_dimension_tensors(
        self.building.cv_size_cm / 100.0,
        self._boundary_cv_mapping,
        self.building.temp.shape,
    )

    (
        self._t_conductivity_left_edge,
        self._t_conductivity_right_edge,
        self._t_conductivity_top_edge,
        self._t_conductivity_bottom_edge,
    ) = get_oriented_conductivity_tensors(
        self.building.conductivity, self._boundary_cv_mapping
    )
    # radiative heat transfer addition
    self.include_radiative_heat_transfer = (
        building.include_radiative_heat_transfer
    )
    # self.interior_wall_mask = building.interior_wall_mask

    # interior mass addition
    self.include_interior_mass = building.include_interior_mass
    if self.include_interior_mass:
      self._initialize_interior_mass_tensors(building)

    # The graph loop covers the configuration without interior mass or
    # radiative exchange. Other configurations keep the eager per-iteration
    # path in Simulator.finite_differences_timestep.
    self._use_graph_fd_loop = (
        use_graph_fd_loop
        and not self.include_interior_mass
        and not self.include_radiative_heat_transfer
        and self._iteration_limit >= 1
    )
    self._cuda_fd = None
    self._cuda_material_refs = None
    self._cuda_fd_fuse_k = cuda_fd_fuse_k
    if self._use_graph_fd_loop:
      self._initialize_graph_fd_loop()
      if use_cuda_fd_kernel:
        self._cuda_fd = self._create_cuda_fd_solver()

  def _create_cuda_fd_solver(self) -> Optional[cuda_fd.CudaFdSolver]:
    """Returns the CUDA FD solver, or None when no GPU or nvcc is available.

    The solver captures building.density and building.heat_capacity, so it is
    rebuilt if either array object is replaced.
    """
    lib = cuda_fd.load_library()
    if lib is None:
      return None
    flags = np.zeros(self.building.temp.shape, dtype=np.uint8)
    flags |= self._t_exerior_temps_mask.numpy().astype(np.uint8)
    for bit, mask in zip((2, 4, 8, 16), self._t_convection_masks):
      flags |= mask.numpy().astype(np.uint8) * np.uint8(bit)
    density = self.building.density
    heat_capacity = self.building.heat_capacity
    try:
      # numpy and tf.convert_to_tensor round float64 to float32 identically.
      solver = cuda_fd.CudaFdSolver(
          lib,
          batch=1,
          shape=self.building.temp.shape,
          iteration_limit=self._iteration_limit,
          convergence_threshold=self._convergence_threshold,
          floor_height_m=self.building.floor_height_cm / 100.0,
          time_step_sec=self._time_step_sec,
          conductivity_left=self._t_conductivity_left_edge.numpy(),
          conductivity_right=self._t_conductivity_right_edge.numpy(),
          conductivity_top=self._t_conductivity_top_edge.numpy(),
          conductivity_bottom=self._t_conductivity_bottom_edge.numpy(),
          u=self._t_u.numpy(),
          v=self._t_v.numpy(),
          density=density,
          heat_capacity=heat_capacity,
          flags=flags,
          fuse_k=self._cuda_fd_fuse_k,
      )
    except (RuntimeError, ValueError) as e:
      logging.warning('CUDA FD solver unavailable, using TF graph: %s', e)
      return None
    self._cuda_material_refs = (density, heat_capacity)
    logging.info(
        'Using the CUDA FD kernel (%d cell classes, %d fused iterations).',
        solver.n_classes,
        self._cuda_fd_fuse_k,
    )
    return solver

  def _initialize_graph_fd_loop(self) -> None:
    """Builds the constants and compiled loop for the graph FD timestep."""
    # Convection masks, so the per-step coefficient tensors equal those from
    # get_oriented_convection_coefficient_tensors without the Python loop.
    (
        t_h_left,
        t_h_right,
        t_h_top,
        t_h_bottom,
    ) = get_oriented_convection_coefficient_tensors(
        1.0, self.building.temp.shape, self._boundary_cv_mapping
    )
    self._t_convection_masks = tuple(
        tf.math.not_equal(t, 0.0)
        for t in (t_h_left, t_h_right, t_h_top, t_h_bottom)
    )
    shape = self.building.temp.shape
    grid = tf.TensorSpec(shape=shape, dtype=tf.float32)
    scalar = tf.TensorSpec(shape=(), dtype=tf.float32)
    self._graph_fd_loop = tf.function(
        self._graph_fd_loop_impl,
        input_signature=[grid, grid, grid, grid, grid, scalar, scalar],
        jit_compile=False,
    )
    # Grappler rewrites change float32 rounding in some CVs, so this function
    # runs with the meta optimizer disabled. The options are thread-local and
    # set only around the call, leaving other tf.functions (the agent) as is.
    ctx = tf_context.context()
    config = ctx.config
    config.allow_soft_placement = True
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    self._graph_fd_call_options = tf_context.FunctionCallOptions(
        config_proto=config
    )

  def _call_graph_fd_loop(self, *args) -> tuple[tf.Tensor, ...]:
    """Calls the graph FD loop with grappler disabled for this call only."""
    thread_local = tf_context.context()._thread_local_data  # pylint: disable=protected-access
    previous = thread_local.function_call_options
    thread_local.function_call_options = self._graph_fd_call_options
    try:
      return self._graph_fd_loop(*args)
    finally:
      thread_local.function_call_options = previous

  def _graph_fd_loop_impl(
      self,
      t_temp_initial: tf.Tensor,
      t_temp_minus: tf.Tensor,
      t_input_q: tf.Tensor,
      t_density: tf.Tensor,
      t_heat_capacity: tf.Tensor,
      t_temp_inf: tf.Tensor,
      t_convection_coefficient: tf.Tensor,
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs the FD iterations of update_temperature_estimates in one graph.

    Each iteration performs the float32 operations of
    update_temperature_estimates in the same order. Terms that do not depend
    on the current estimate are computed once per timestep instead of once per
    iteration, which yields the same values.

    Args:
      t_temp_initial: First temperature estimate, float32.
      t_temp_minus: Building temperature at the previous timestep, float32.
      t_input_q: External heat source per CV, float32.
      t_density: Density per CV, float32.
      t_heat_capacity: Heat capacity per CV, float32.
      t_temp_inf: Ambient temperature, float32.
      t_convection_coefficient: Air convection coefficient, float32.

    Returns:
      The final estimate, the number of iterations run, and the max_delta of
      every iteration run (unused entries are zero).
    """
    zero = tf.zeros_like(t_temp_initial)
    t_h_left, t_h_right, t_h_top, t_h_bottom = (
        tf.where(mask, t_convection_coefficient, zero)
        for mask in self._t_convection_masks
    )
    t_z = tf.constant(self.building.floor_height_cm / 100.0, dtype=tf.float32)
    t_delta_t = tf.constant(self._time_step_sec, dtype=tf.float32)
    t_u = self._t_u
    t_v = self._t_v

    t_uz = tf.scalar_mul(t_z, t_u)
    t_vz = tf.scalar_mul(t_z, t_v)
    t_k1_div_u = tf.math.divide(self._t_conductivity_left_edge, t_u)
    t_k3_div_u = tf.math.divide(self._t_conductivity_right_edge, t_u)
    t_k2_div_v = tf.math.divide(self._t_conductivity_bottom_edge, t_v)
    t_k4_div_v = tf.math.divide(self._t_conductivity_top_edge, t_v)

    # Denominator, as in _get_denominator (dt4 is zero without interior mass).
    dt1 = tf.math.add(t_k1_div_u, t_k3_div_u)
    dt1 = tf.math.add(dt1, t_h_left)
    dt1 = tf.math.add(dt1, t_h_right)
    dt1 = tf.math.multiply(t_vz, dt1)
    dt2 = tf.math.add(t_k2_div_v, t_k4_div_v)
    dt2 = tf.math.add(dt2, t_h_bottom)
    dt2 = tf.math.add(dt2, t_h_top)
    dt2 = tf.math.multiply(t_uz, dt2)
    storage = tf.math.multiply(t_density, t_u)
    storage = tf.math.multiply(storage, t_v)
    storage = tf.math.multiply(storage, t_heat_capacity)
    storage = tf.scalar_mul(t_z, storage)
    dt3 = tf.math.divide(storage, t_delta_t)
    t_denom = tf.math.add(dt1, dt2)
    t_denom = tf.math.add(t_denom, dt3)
    t_denom = tf.math.add(t_denom, tf.zeros_like(dt3))

    # Estimate-independent numerator terms, as in _get_numerator.
    t_h_left_tinf = tf.math.scalar_mul(t_temp_inf, t_h_left)
    t_h_right_tinf = tf.math.scalar_mul(t_temp_inf, t_h_right)
    t_h_above_tinf = tf.math.scalar_mul(t_temp_inf, t_h_top)
    t_h_below_tinf = tf.math.scalar_mul(t_temp_inf, t_h_bottom)
    nt3 = tf.math.multiply(storage, t_temp_minus)
    nt3 = tf.math.divide(nt3, t_delta_t)

    t_exterior_temp = tf.fill(tf.shape(t_temp_initial), t_temp_inf)
    threshold = tf.constant(self._convergence_threshold, dtype=tf.float64)
    limit = self._iteration_limit

    def body(i, t_temp, _, deltas):
      # Same values as shift_tensor_{left,right,down,up}, without the masks.
      t_temp_left = tf.concat(
          [t_temp[:, 1:], tf.fill([t_temp.shape[0], 1], t_temp_inf)], axis=1
      )
      t_temp_right = tf.concat(
          [tf.fill([t_temp.shape[0], 1], t_temp_inf), t_temp[:, :-1]], axis=1
      )
      t_temp_above = tf.concat(
          [tf.fill([1, t_temp.shape[1]], t_temp_inf), t_temp[:-1, :]], axis=0
      )
      t_temp_below = tf.concat(
          [t_temp[1:, :], tf.fill([1, t_temp.shape[1]], t_temp_inf)], axis=0
      )

      nt1 = tf.math.add(
          tf.math.multiply(t_k1_div_u, t_temp_left),
          tf.math.multiply(t_k3_div_u, t_temp_right),
      )
      nt1 = tf.math.add(nt1, t_h_left_tinf)
      nt1 = tf.math.add(nt1, t_h_right_tinf)
      nt1 = tf.math.multiply(t_vz, nt1)
      nt2 = tf.math.add(
          tf.math.multiply(t_k2_div_v, t_temp_below),
          tf.math.multiply(t_k4_div_v, t_temp_above),
      )
      nt2 = tf.math.add(nt2, t_h_below_tinf)
      nt2 = tf.math.add(nt2, t_h_above_tinf)
      nt2 = tf.math.multiply(t_uz, nt2)
      t_numer = tf.math.add(nt1, nt2)
      t_numer = tf.math.add(t_numer, nt3)
      t_numer = tf.math.add(t_numer, t_input_q)
      t_numer = tf.math.add(t_numer, zero)
      t_numer = tf.math.add(t_numer, zero)

      t_estimate = tf.math.divide(t_numer, t_denom)
      t_estimate = tf.where(
          self._t_exerior_temps_mask, t_exterior_temp, t_estimate
      )
      max_delta = tf.math.reduce_max(
          tf.math.abs(tf.math.subtract(t_estimate, t_temp))
      )
      deltas = tf.tensor_scatter_nd_update(deltas, [[i]], [max_delta])
      # numpy compares the float32 max_delta with the Python float threshold
      # in float64.
      converged = tf.cast(max_delta, tf.float64) <= threshold
      return i + 1, t_estimate, converged, deltas

    def cond(i, unused_t_temp, converged, unused_deltas):
      return tf.logical_and(i < limit, tf.logical_not(converged))

    n_iter, t_temp, _, deltas = tf.while_loop(
        cond,
        body,
        (
            tf.constant(0),
            t_temp_initial,
            tf.constant(False),
            tf.zeros([max(limit, 1)], dtype=tf.float32),
        ),
    )
    return t_temp, n_iter, deltas

  def finite_differences_timestep(
      self, *, ambient_temperature: float, convection_coefficient: float
  ) -> bool:
    """Runs Simulator.finite_differences_timestep on the GPU or as a TF graph.

    Results, warnings, and the return value match the parent method. Uses the
    CUDA kernel when available, else one compiled TF graph. Falls back to the
    parent when interior mass or radiative exchange is enabled, or when
    use_graph_fd_loop is False.

    Args:
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).

    Returns:
      Whether or not there was convergence before iteration_limit was reached.
    """
    if not self._use_graph_fd_loop:
      return super().finite_differences_timestep(
          ambient_temperature=ambient_temperature,
          convection_coefficient=convection_coefficient,
      )

    building = self.building
    if self._cuda_fd is not None and (
        self._cuda_material_refs[0] is not building.density
        or self._cuda_material_refs[1] is not building.heat_capacity
    ):
      self._cuda_fd = self._create_cuda_fd_solver()
    if self._cuda_fd is not None:
      # numpy and tf.convert_to_tensor round float64 to float32 identically.
      out, n_iter_arr, deltas_arr = self._cuda_fd.step(
          np.asarray(building.temp, dtype=np.float32)[None],
          np.asarray(building.input_q, dtype=np.float32)[None],
          np.array([ambient_temperature], dtype=np.float32),
          np.array([convection_coefficient], dtype=np.float32),
      )
      n_iter = int(n_iter_arr[0])
      deltas = deltas_arr[0].copy()
      # A new array each step, as t_temp.numpy() gives in the graph path.
      new_temp = out[0].copy()
    else:
      t_temp, n_iter, deltas = self._call_graph_fd_loop(  # pylint: disable=unpacking-non-sequence
          tf.convert_to_tensor(building.temp.copy(), dtype=tf.float32),
          tf.convert_to_tensor(building.temp, dtype=tf.float32),
          tf.convert_to_tensor(building.input_q, dtype=tf.float32),
          tf.convert_to_tensor(building.density, dtype=tf.float32),
          tf.convert_to_tensor(building.heat_capacity, dtype=tf.float32),
          tf.constant(ambient_temperature, dtype=tf.float32),
          tf.constant(convection_coefficient, dtype=tf.float32),
      )
      n_iter = int(n_iter)
      deltas = deltas.numpy()
      new_temp = t_temp.numpy()

    if n_iter >= self._iteration_warning:
      logging.warning(
          'Step %d, not converged in %d steps, max_delta = %3.3f',
          self._iteration_warning - 1,
          self._iteration_warning,
          deltas[self._iteration_warning - 1],
      )
    converged_successfully = n_iter > 0 and (
        deltas[n_iter - 1] <= self._convergence_threshold
    )
    if not converged_successfully:
      logging.warning(
          'Max iteration count reached, max_delta = %3.3f',
          deltas[n_iter - 1] if n_iter > 0 else float('nan'),
      )

    building.temp = new_temp
    return converged_successfully

  def _get_tensor_exterior_mask(
      self, building: building_py.Building
  ) -> tf.Tensor:
    """Returns a binary tensor mask of all CVs that are exterior = True."""
    exterior_cv_mapping = get_cv_mapping(
        building.neighbors, position_criterion=CVPositionType.EXTERIOR
    )
    exterior_mask = np.full(building.temp.shape, False)
    for i, j in exterior_cv_mapping:
      exterior_mask[i][j] = True
    return tf.convert_to_tensor(exterior_mask)

  def _initialize_interior_mass_tensors(
      self, building: building_py.FloorPlanBasedBuilding
  ) -> None:
    """Initializes tensors for interior mass calculations."""
    # Convert interior mass properties to tensors
    self._t_interior_mass_mask = tf.convert_to_tensor(
        building.interior_mass_mask, dtype=tf.bool
    )
    self._t_interior_mass_conductivity = tf.convert_to_tensor(
        building.interior_mass_conductivity, dtype=tf.float32
    )
    self._t_interior_mass_heat_capacity = tf.convert_to_tensor(
        building.interior_mass_heat_capacity, dtype=tf.float32
    )
    self._t_interior_mass_density = tf.convert_to_tensor(
        building.interior_mass_density, dtype=tf.float32
    )

    # Calculate t_0_mass = (rho_mass * C_mass * z^2) / (k_mass * delta_t)
    # z is the floor height in meters (characteristic length for heat exchange)
    z = building.floor_height_cm / 100.0

    # Compute t_0_mass for each CV (will be 0 where there's no interior mass)
    self._t_0_mass = tf.where(
        self._t_interior_mass_mask,
        (
            self._t_interior_mass_density
            * self._t_interior_mass_heat_capacity
            * z
            * z
        )
        / (self._t_interior_mass_conductivity * self._time_step_sec),
        tf.constant(0.0, dtype=tf.float32),
    )

  def update_interior_mass_temperatures(
      self, air_temperature_estimates: np.ndarray
  ) -> tuple[np.ndarray, float]:
    r"""Tensorized version of interior mass temperature update.

    Overrides the parent class's iterative implementation with a tensor-based
    approach for efficiency. The heat exchange occurs through the vertical
    direction (height z) of the control volume.

    Equations:
    --------------------
    After the air CV temperatures converge, update the interior mass
    temperatures using element-wise tensor operations:

    $$T_{\text{mass}} =
      \frac{T+t_{0,\text{mass}}\odot T_{\text{mass}}^{(-)}}{1+t_{0,\text{mass}}}
    $$

    where $\odot$ represents element-wise multiplication and the division is
    element-wise (broadcasting the scalar denominator).

    The temporal parameter tensor is defined as:

    $$t_{0,\text{mass},i,j} = \begin{cases}
    \frac{\rho_{\text{mass}}c_{\text{mass}}z^2}{k_{\text{mass}}\Delta t}
    & \text{if CV has interior mass} \\
    0 & \text{otherwise}
    \end{cases}$$

    This formulation is consistent with the air CV energy balance where the
    interior mass coupling term is $\frac{k_{\text{mass}} u v}{z}
    (T_{\text{mass},i,j} - T_{i,j})$.

    Nomenclature and Units:
    -----------------------
    - $T$: Converged air temperature tensor at new time step [K]
    - $T_{\text{mass}}$: Interior mass temperature tensor at new time step [K]
    - $T_{\text{mass}}^{(-)}$: Interior mass temperature tensor at previous
      time step [K]
    - $t_{0,\text{mass}}$: Temporal parameter tensor for interior mass
      [dimensionless]
    - $k_{\text{mass}}$: Thermal conductivity of interior mass
      [$\mathrm{W/(m \cdot K)}$]
    - $\rho_{\text{mass}}$: Density of interior mass [$\mathrm{kg/m^3}$]
    - $c_{\text{mass}}$: Specific heat capacity of interior mass
      [$\mathrm{J/(kg \cdot K)}$]
    - $z$: CV height (floor height), characteristic length for heat exchange
      [$\mathrm{m}$]
    - $\Delta t$: Time step [$\mathrm{s}$]

    Args:
      air_temperature_estimates: Current air temperature estimates for each CV.

    Returns:
      Tuple of (updated interior mass temperatures, maximum temperature change)
    """
    if not self.include_interior_mass:
      return self.building.interior_mass_temp.copy(), 0.0

    # Convert air temperatures to tensor
    t_air_temp = tf.convert_to_tensor(
        air_temperature_estimates, dtype=tf.float32
    )

    # Convert previous interior mass temperature to tensor
    t_temp_mass_prev = tf.convert_to_tensor(
        self.building.interior_mass_temp, dtype=tf.float32
    )

    # Calculate numerator: T + t_0_mass * T_mass^(-)
    numerator = tf.math.add(
        t_air_temp, tf.math.multiply(self._t_0_mass, t_temp_mass_prev)
    )

    # Calculate denominator: 1 + t_0_mass
    denominator = tf.math.add(
        tf.constant(1.0, dtype=tf.float32), self._t_0_mass
    )

    # Update interior mass temperature
    t_temp_mass_new = tf.math.divide(numerator, denominator)

    # Apply mask to only update where interior mass exists
    t_temp_mass_new = tf.where(
        self._t_interior_mass_mask, t_temp_mass_new, t_temp_mass_prev
    )

    # Calculate maximum change for convergence checking
    t_delta = tf.math.subtract(t_temp_mass_new, t_temp_mass_prev)
    max_delta = np.max(tf.math.abs(t_delta))

    # Return as numpy arrays
    return t_temp_mass_new.numpy(), max_delta

  def update_temperature_estimates(
      self,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> tuple[np.ndarray, float]:
    r"""Iterates across all CVs and updates the temperature estimate.

    Corner and edge CVs are exposed to thermal exchange with the ambient air
    through convection. This tensorized implementation overrides the parent
    class's iterative approach for computational efficiency.

    Equations:
    --------------------
    The tensorized heat balance equation for air CV with interior mass
    coupling is:

    $$\begin{multline}
      T = \left[Q_x + Vz\left[K_1U^{-1}T_1 + H_1T_\infty + K_3U^{-1}T_3 +
        H_3T_\infty\right] \right. \\
      \left. + Uz\left[K_2V^{-1}T_2 + H_2T_\infty + K_4V^{-1}T_4 +
        H_4T_\infty\right] \right. \\
      \left. + K_{\text{mass}}UVz^{-1}T_{\text{mass}} + Q_{\text{lwx}} +
        \frac{C\rho UVz}{\Delta t}T^{(-)}\right] \\
      \cdot \left[Vz\left[K_1U^{-1} + H_1 + K_3U^{-1} + H_3\right] +
        Uz\left[K_2V^{-1} + H_2 + K_4V^{-1} + H_4\right] \right. \\
      \left. + K_{\text{mass}}UVz^{-1} +
        \frac{C\rho UVz}{\Delta t}\right]^{-1}
      \end{multline}$$

    where the shifted temperature tensors represent neighboring CVs:
    - $T_1 = \text{shift}(T, \text{LEFT})$
    - $T_2 = \text{shift}(T, \text{DOWN})$
    - $T_3 = \text{shift}(T, \text{RIGHT})$
    - $T_4 = \text{shift}(T, \text{UP})$

    Nomenclature and Units:
    -----------------------
    - $T$: Air temperature tensor at new time step [K]
    - $T^{(-)}$: Air temperature tensor at previous time step [K]
    - $T_1, T_2, T_3, T_4$: Temperature tensors of neighboring CVs
      (left, down, right, up) [K]
    - $T_{\text{mass}}$: Interior mass temperature tensor [K]
    - $T_\infty$: Ambient temperature (scalar) [K]
    - $Q_x$: External heat source tensor [$\mathrm{W}$]
    - $Q_{\text{lwx}}$: Longwave radiative exchange tensor [$\mathrm{W}$]
    - $K_1, K_2, K_3, K_4$: Thermal conductivity tensors for left, down,
      right, up faces [$\mathrm{W/(m \cdot K)}$]
    - $K_{\text{mass}}$: Interior mass conductivity tensor
      [$\mathrm{W/(m \cdot K)}$]
    - $H_1, H_2, H_3, H_4$: Convection coefficient tensors for boundary CVs
      [$\mathrm{W/(m^2 \cdot K)}$]
    - $U, V$: CV dimensions in x and y directions [$\mathrm{m}$]
    - $z$: CV height (floor height) [$\mathrm{m}$]
    - $C$: Specific heat capacity tensor [$\mathrm{J/(kg \cdot K)}$]
    - $\rho$: Density tensor [$\mathrm{kg/m^3}$]
    - $\Delta t$: Time step [$\mathrm{s}$]

    References:
    -----------
    - Equation 22 derived in go/smart-buildings-simulator-design

    Args:
      temperature_estimates: Current temperature estimate for each CV, will be
        updated with new values.
      ambient_temperature: Current temperature in K of external air.
      convection_coefficient: Current wind convection coefficient (W/m2/K).

    Returns:
      Maximum difference in temperture_estimates across all CVs before and after
      operation.
    """

    def _get_input_tensors(
        building,
    ) -> tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
      """Returns the input matrices as tensors."""
      # Convert a bunch of numpy arrays into TF tensors.
      t_temp = tf.convert_to_tensor(temperature_estimates, dtype=tf.float32)
      t_temp_old = tf.convert_to_tensor(temperature_estimates, dtype=tf.float32)
      t_temp_minus = tf.convert_to_tensor(building.temp, dtype=tf.float32)
      t_input_q = tf.convert_to_tensor(building.input_q, dtype=tf.float32)
      t_density = tf.convert_to_tensor(building.density, dtype=tf.float32)
      t_heat_capacity = tf.convert_to_tensor(
          building.heat_capacity, dtype=tf.float32
      )
      t_z = tf.constant(building.floor_height_cm / 100.0, dtype=tf.float32)
      if self.include_radiative_heat_transfer:
        t_ifa_inv = tf.convert_to_tensor(building.ifa_inv, dtype=tf.float32)
        # For radiative heat transfer, we need to combine interior wall and
        # interior mass temperatures if interior mass is enabled
        if self.include_interior_mass:
          interior_mask_all = (
              building.interior_wall_mask | building.interior_mass_mask
          )
          temperature_estimates_temp = np.zeros_like(temperature_estimates)
          temperature_estimates_temp[building.interior_mass_mask] = (
              building.interior_mass_temp[building.interior_mass_mask]
          )
          temperature_estimates_temp[building.interior_wall_mask] = (
              temperature_estimates[building.interior_wall_mask]
          )
          t_temp_interior_wall = tf.convert_to_tensor(
              temperature_estimates_temp[interior_mask_all], dtype=tf.float32
          )
        else:
          t_temp_interior_wall = tf.convert_to_tensor(
              temperature_estimates[building.interior_wall_mask],
              dtype=tf.float32,
          )
        # Ensure t_temp_interior_wall is a column vector for matrix
        # multiplication
        t_temp_interior_wall = tf.reshape(t_temp_interior_wall, [-1, 1])
      else:
        # Create minimal zero tensors with appropriate shapes
        # These won't be used when radiative heat transfer is disabled
        t_ifa_inv = tf.zeros((1, 1), dtype=tf.float32)  # Minimal shape
        t_temp_interior_wall = tf.zeros((1,), dtype=tf.float32)  # Minimal shape

      # Interior mass temperature tensor
      if self.include_interior_mass:
        t_temp_mass = tf.convert_to_tensor(
            building.interior_mass_temp, dtype=tf.float32
        )
      else:
        t_temp_mass = tf.zeros((1, 1), dtype=tf.float32)  # Minimal shape

      return (
          t_temp,
          t_temp_old,
          t_temp_minus,
          t_input_q,
          t_density,
          t_heat_capacity,
          t_z,
          t_ifa_inv,
          t_temp_interior_wall,
          t_temp_mass,
      )

    def _get_neighbor_temps(
        t_temp: tf.Tensor, ambient_temperature: float
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
      """Creates left, right, up, down neighbor temp matrices."""

      # Create shifted tensor to be able to evaluate neighbors in the equation.
      t_temp_left = shift_tensor_left(t_temp, padding_value=ambient_temperature)

      t_temp_right = shift_tensor_right(
          t_temp, padding_value=ambient_temperature
      )

      t_temp_above = shift_tensor_down(
          t_temp, padding_value=ambient_temperature
      )

      t_temp_below = shift_tensor_up(t_temp, padding_value=ambient_temperature)

      return (t_temp_left, t_temp_right, t_temp_above, t_temp_below)

    def _get_denominator(
        t_k1_div_u: tf.Tensor,
        t_k3_div_u: tf.Tensor,
        t_convection_left_edge: tf.Tensor,
        t_convection_right_edge: tf.Tensor,
        t_convection_top_edge: tf.Tensor,
        t_convection_bottom_edge: tf.Tensor,
        t_vz: tf.Tensor,
        t_k2_div_v: tf.Tensor,
        t_k4_div_v: tf.Tensor,
        t_uz: tf.Tensor,
        t_density: tf.Tensor,
        t_heat_capacity: tf.Tensor,
        t_z: tf.Tensor,
        t_delta_t: tf.Tensor,
    ) -> tf.Tensor:
      """Returns the denominator matrix from Eqn 22 as a tensor."""

      # Compute conductivity/conduction transfer terms on the v-z surface.
      dt1 = tf.math.add(t_k1_div_u, t_k3_div_u)
      dt1 = tf.math.add(dt1, t_convection_left_edge)
      dt1 = tf.math.add(dt1, t_convection_right_edge)
      dt1 = tf.math.multiply(t_vz, dt1)

      # Compute conductivity/conduction transfer terms on the u-z surface.
      dt2 = tf.math.add(t_k2_div_v, t_k4_div_v)
      dt2 = tf.math.add(dt2, t_convection_bottom_edge)
      dt2 = tf.math.add(dt2, t_convection_top_edge)
      dt2 = tf.math.multiply(t_uz, dt2)

      # Create the thermal absorption (storage) term: C * rho * U * V * z / dt.
      dt3 = tf.math.multiply(t_density, self._t_u)
      dt3 = tf.math.multiply(dt3, self._t_v)
      dt3 = tf.math.multiply(dt3, t_heat_capacity)
      dt3 = tf.scalar_mul(t_z, dt3)
      dt3 = tf.math.divide(dt3, t_delta_t)

      # Add interior mass coupling term: K_mass * U * V / Z
      dt4 = tf.zeros_like(dt3)
      if self.include_interior_mass:
        dt4 = tf.math.multiply(self._t_interior_mass_conductivity, self._t_u)
        dt4 = tf.math.multiply(dt4, self._t_v)
        dt4 = tf.math.divide(dt4, t_z)

      # Sum up u-z, u-v surface transfer, absorption, and interior mass terms.
      t_denom = tf.math.add(dt1, dt2)
      t_denom = tf.math.add(t_denom, dt3)
      t_denom = tf.math.add(t_denom, dt4)
      return t_denom

    def _get_numerator(
        t_k1_div_u: tf.Tensor,
        t_k3_div_u: tf.Tensor,
        t_convection_left_edge: tf.Tensor,
        t_convection_right_edge: tf.Tensor,
        t_convection_top_edge: tf.Tensor,
        t_convection_bottom_edge: tf.Tensor,
        t_vz: tf.Tensor,
        t_k2_div_v: tf.Tensor,
        t_k4_div_v: tf.Tensor,
        t_uz: tf.Tensor,
        t_density: tf.Tensor,
        t_heat_capacity: tf.Tensor,
        t_z: tf.Tensor,
        t_delta_t: tf.Tensor,
        t_temp_left: tf.Tensor,
        t_temp_right: tf.Tensor,
        t_temp_above: tf.Tensor,
        t_temp_below: tf.Tensor,
        t_temp_inf: tf.Tensor,
        t_input_q: tf.Tensor,
        t_temp_minus: tf.Tensor,
        t_ifa_inv: tf.Tensor,
        t_temp_interior_wall: tf.Tensor,
        t_temp_mass: tf.Tensor,
    ) -> tf.Tensor:
      """Returns the numerator matrix from Eqn 22 as a tensor."""

      # Compute numerator's conductivity transfer terms.
      t_k1_div_u_temp_left = tf.math.multiply(t_k1_div_u, t_temp_left)
      t_k3_div_u_temp_right = tf.math.multiply(t_k3_div_u, t_temp_right)
      t_k2_div_v_temp_below = tf.math.multiply(t_k2_div_v, t_temp_below)
      t_k4_div_v_temp_above = tf.math.multiply(t_k4_div_v, t_temp_above)

      # Compute numerator's convection transfer terms.
      t_h_left_tinf = tf.math.scalar_mul(t_temp_inf, t_convection_left_edge)
      t_h_right_tinf = tf.math.scalar_mul(t_temp_inf, t_convection_right_edge)
      t_h_above_tinf = tf.math.scalar_mul(t_temp_inf, t_convection_top_edge)
      t_h_below_tinf = tf.math.scalar_mul(t_temp_inf, t_convection_bottom_edge)

      # Merge the conduction/convection transfer terms across the v-z surfaces.
      nt1 = tf.math.add(t_k1_div_u_temp_left, t_k3_div_u_temp_right)
      nt1 = tf.math.add(nt1, t_h_left_tinf)
      nt1 = tf.math.add(nt1, t_h_right_tinf)
      nt1 = tf.math.multiply(t_vz, nt1)

      # Merge the conduction/convection transfer terms across the u-z surfaces.
      nt2 = tf.math.add(t_k2_div_v_temp_below, t_k4_div_v_temp_above)
      nt2 = tf.math.add(nt2, t_h_below_tinf)
      nt2 = tf.math.add(nt2, t_h_above_tinf)
      nt2 = tf.math.multiply(t_uz, nt2)

      # Create the thermal absorption (storage) term:
      # C * rho * U * V * z / dt * T^(-).
      nt3 = tf.math.multiply(t_density, self._t_u)
      nt3 = tf.math.multiply(nt3, self._t_v)
      nt3 = tf.math.multiply(nt3, t_heat_capacity)
      nt3 = tf.scalar_mul(t_z, nt3)
      nt3 = tf.math.multiply(nt3, t_temp_minus)
      nt3 = tf.math.divide(nt3, t_delta_t)

      # add ratdative heat transfer sigma*ifa_inv@(T-)^4
      nt4 = tf.zeros_like(t_temp_minus)
      if self.include_radiative_heat_transfer:
        sigma = tf.constant(5.67e-8, dtype=tf.float32)
        t_temp_interior_wall_4 = tf.math.pow(t_temp_interior_wall, 4)
        # Ensure both tensors have the same dtype for matrix multiplication
        nt4_temp = tf.linalg.matmul(t_ifa_inv, t_temp_interior_wall_4)
        nt4_temp = tf.math.multiply(nt4_temp, sigma)

        # Use tensor_scatter_nd_update to update specific indices
        indices = tf.where(self.building.lwx_index >= 0)
        # Extract the specific elements from nt4_temp and flatten to match nt4
        # shape
        updates = tf.gather(
            tf.squeeze(
                nt4_temp
            ),  # Remove the extra dimension from [26,1] to [26]
            self.building.lwx_index[self.building.lwx_index >= 0],
        )
        nt4 = tf.tensor_scatter_nd_update(nt4, indices, updates)

      # Add interior mass coupling term: K_mass * U * V / Z * T_mass
      nt5 = tf.zeros_like(t_temp_minus)
      if self.include_interior_mass:
        nt5 = tf.math.multiply(self._t_interior_mass_conductivity, self._t_u)
        nt5 = tf.math.multiply(nt5, self._t_v)
        nt5 = tf.math.multiply(nt5, t_temp_mass)
        nt5 = tf.math.divide(nt5, t_z)

      # Add the u-z, u-v surface transfer, absorption, external source,
      #  and interior mass terms.
      t_numer = tf.math.add(nt1, nt2)
      t_numer = tf.math.add(t_numer, nt3)
      t_numer = tf.math.add(t_numer, t_input_q)
      t_numer = tf.math.add(t_numer, nt4)
      t_numer = tf.math.add(t_numer, nt5)
      return t_numer

    # Get the inputs to the equation as Tensors from the building.
    (
        t_temp,
        t_temp_old,
        t_temp_minus,
        t_input_q,
        t_density,
        t_heat_capacity,
        t_z,
        t_ifa_inv,
        t_temp_interior_wall,
        t_temp_mass,
    ) = _get_input_tensors(self.building)

    (
        t_convection_left_edge,
        t_convection_right_edge,
        t_convection_top_edge,
        t_convection_bottom_edge,
    ) = get_oriented_convection_coefficient_tensors(
        convection_coefficient,
        self.building.temp.shape,
        self._boundary_cv_mapping,
    )

    # Create shifted tensor to be able to evaluate neighbors in the equation.
    t_temp_left, t_temp_right, t_temp_above, t_temp_below = _get_neighbor_temps(
        t_temp, ambient_temperature
    )

    # Get the ambinet temperature as a tensor.
    t_temp_inf = tf.constant(ambient_temperature, dtype=tf.float32)

    # Convert the timestep input to tensor.
    t_delta_t = tf.constant(self._time_step_sec, dtype=tf.float32)

    # Create surface area tensors: horizontal u or v x vertical z dim.
    t_uz = tf.scalar_mul(t_z, self._t_u)
    t_vz = tf.scalar_mul(t_z, self._t_v)

    # Calculate the denominator terms.
    t_k1_div_u = tf.math.divide(self._t_conductivity_left_edge, self._t_u)
    t_k3_div_u = tf.math.divide(self._t_conductivity_right_edge, self._t_u)
    t_k2_div_v = tf.math.divide(self._t_conductivity_bottom_edge, self._t_v)
    t_k4_div_v = tf.math.divide(self._t_conductivity_top_edge, self._t_v)

    t_denom = _get_denominator(
        t_k1_div_u,
        t_k3_div_u,
        t_convection_left_edge,
        t_convection_right_edge,
        t_convection_top_edge,
        t_convection_bottom_edge,
        t_vz,
        t_k2_div_v,
        t_k4_div_v,
        t_uz,
        t_density,
        t_heat_capacity,
        t_z,
        t_delta_t,
    )

    # Calculate the numerator terms
    t_numer = _get_numerator(
        t_k1_div_u,
        t_k3_div_u,
        t_convection_left_edge,
        t_convection_right_edge,
        t_convection_top_edge,
        t_convection_bottom_edge,
        t_vz,
        t_k2_div_v,
        t_k4_div_v,
        t_uz,
        t_density,
        t_heat_capacity,
        t_z,
        t_delta_t,
        t_temp_left,
        t_temp_right,
        t_temp_above,
        t_temp_below,
        t_temp_inf,
        t_input_q,
        t_temp_minus,
        t_ifa_inv,
        t_temp_interior_wall,
        t_temp_mass,
    )

    # Finally, perform an elementwise division - not a matrix inversion.
    t_temperature_estimates = tf.math.divide(t_numer, t_denom)

    # The tensor operation potentially altered the exterior air conditions,
    # so we need to reset exterior CVs to the exterior air conditioners.
    t_temperature_estimates = apply_exterior_temps(
        t_temperature_estimates, t_temp_inf, self._t_exerior_temps_mask
    )

    t_delta = tf.math.subtract(t_temperature_estimates, t_temp_old)

    # Note: Interior mass temperatures are updated by the parent class's
    # finite_differences_timestep() method after this function returns.
    # Do NOT update them here to avoid double-updating.

    return t_temperature_estimates.numpy(), np.max(tf.math.abs(t_delta))
