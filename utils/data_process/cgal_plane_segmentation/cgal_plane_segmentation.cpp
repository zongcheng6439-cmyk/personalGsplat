#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/io.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing_on_point_set.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Vector = Kernel::Vector_3;
using Plane = Kernel::Plane_3;
using Point_set = CGAL::Point_set_3<Point>;
using Point_map = Point_set::Point_map;
using Normal_map = Point_set::Vector_map;
using Neighbor_query = CGAL::Shape_detection::Point_set::K_neighbor_query<
    Kernel, Point_set, Point_map>;
using Region_type = CGAL::Shape_detection::Point_set::Least_squares_plane_fit_region<
    Kernel, Point_set, Point_map, Normal_map>;
using Sorting = CGAL::Shape_detection::Point_set::Least_squares_plane_fit_sorting<
    Kernel, Point_set, Neighbor_query, Point_map>;
using Region_growing = CGAL::Shape_detection::Region_growing<
    Point_set, Neighbor_query, Region_type, Sorting::Seed_map>;
using Region = std::vector<std::size_t>;

struct Options {
  std::string input;
  std::string output;
  double voxel_size = 0.03;
  std::size_t normal_k = 24;
  std::size_t region_k = 18;
  double distance = 0.06;
  double normal_angle = 20.0;
  std::size_t min_points = 300;
  double max_rmse = 0.04;
  double axis_angle = 30.0;
  std::string anchor_output;
  double anchor_voxel_size = 0.1;
  double anchor_min_plane_ratio = 0.5;
  double anchor_min_normal_coherence = 0.9;
  bool overwrite = false;
};

[[noreturn]] void usage(const char* program, int code) {
  std::ostream& out = code == 0 ? std::cout : std::cerr;
  out << "Usage: " << program << " --input INPUT.ply --output OUTPUT.ply [options]\n"
      << "  --voxel-size M       Detection downsample size (default 0.03)\n"
      << "  --normal-k N         PCA normal neighbors (default 24)\n"
      << "  --region-k N         Region-growing neighbors (default 18)\n"
      << "  --distance M         Maximum point-to-plane distance (default 0.06)\n"
      << "  --normal-angle DEG   Region normal tolerance (default 20)\n"
      << "  --min-points N       Minimum downsampled points per plane (default 300)\n"
      << "  --max-rmse M         Maximum fitted plane RMSE (default 0.04)\n"
      << "  --axis-angle DEG     Ground/wall tilt tolerance (default 30)\n"
      << "  --anchor-output PLY  Voxel-anchor output (default OUTPUT_anchors.ply)\n"
      << "  --anchor-voxel-size M  Anchor voxel size (default 0.1)\n"
      << "  --anchor-min-plane-ratio X  Minimum planar-point ratio (default 0.5)\n"
      << "  --anchor-min-normal-coherence X  Minimum normal coherence (default 0.9)\n"
      << "  --overwrite          Replace an existing output\n";
  std::exit(code);
}

template <typename T>
T parse_number(const std::string& text, const std::string& name) {
  std::size_t used = 0;
  if constexpr (std::is_same_v<T, std::size_t>) {
    const auto value = std::stoull(text, &used);
    if (used != text.size()) throw std::invalid_argument(name);
    return static_cast<std::size_t>(value);
  } else {
    const auto value = std::stod(text, &used);
    if (used != text.size()) throw std::invalid_argument(name);
    return static_cast<T>(value);
  }
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string name = argv[i];
    if (name == "--help" || name == "-h") usage(argv[0], 0);
    if (name == "--overwrite") {
      options.overwrite = true;
      continue;
    }
    if (i + 1 >= argc) usage(argv[0], 2);
    const std::string value = argv[++i];
    if (name == "--input") options.input = value;
    else if (name == "--output") options.output = value;
    else if (name == "--voxel-size") options.voxel_size = parse_number<double>(value, name);
    else if (name == "--normal-k") options.normal_k = parse_number<std::size_t>(value, name);
    else if (name == "--region-k") options.region_k = parse_number<std::size_t>(value, name);
    else if (name == "--distance") options.distance = parse_number<double>(value, name);
    else if (name == "--normal-angle") options.normal_angle = parse_number<double>(value, name);
    else if (name == "--min-points") options.min_points = parse_number<std::size_t>(value, name);
    else if (name == "--max-rmse") options.max_rmse = parse_number<double>(value, name);
    else if (name == "--axis-angle") options.axis_angle = parse_number<double>(value, name);
    else if (name == "--anchor-output") options.anchor_output = value;
    else if (name == "--anchor-voxel-size") options.anchor_voxel_size = parse_number<double>(value, name);
    else if (name == "--anchor-min-plane-ratio") options.anchor_min_plane_ratio = parse_number<double>(value, name);
    else if (name == "--anchor-min-normal-coherence") options.anchor_min_normal_coherence = parse_number<double>(value, name);
    else throw std::invalid_argument("unknown argument: " + name);
  }
  if (options.input.empty() || options.output.empty()) usage(argv[0], 2);
  if (options.voxel_size < 0 || options.normal_k < 3 || options.region_k < 3 ||
      options.distance <= 0 || options.min_points < 3 || options.max_rmse <= 0 ||
      options.normal_angle <= 0 || options.normal_angle >= 90 ||
      options.axis_angle <= 0 || options.axis_angle >= 45 ||
      options.anchor_voxel_size <= 0 || options.anchor_min_plane_ratio < 0 ||
      options.anchor_min_plane_ratio > 1 ||
      options.anchor_min_normal_coherence < 0 ||
      options.anchor_min_normal_coherence > 1) {
    throw std::invalid_argument("invalid segmentation parameter");
  }
  return options;
}

struct Fitted_plane {
  Vector normal;
  Point centroid;
  double rmse;
  std::uint8_t type;
};

Fitted_plane fit_plane(const Point_set& points, const Region& region,
                       double axis_angle) {
  std::vector<Point> samples;
  samples.reserve(region.size());
  for (const std::size_t offset : region)
    samples.push_back(points.point(*(points.begin() + offset)));

  Plane plane;
  Point centroid;
  CGAL::linear_least_squares_fitting_3(
      samples.begin(), samples.end(), plane, centroid, CGAL::Dimension_tag<0>());
  Vector normal = plane.orthogonal_vector();
  const double length = std::sqrt(normal.squared_length());
  normal = normal / length;
  const double components[3] = {normal.x(), normal.y(), normal.z()};
  const int dominant = std::max_element(
      std::begin(components), std::end(components),
      [](double a, double b) { return std::abs(a) < std::abs(b); }) -
      std::begin(components);
  if (components[dominant] < 0) normal = -normal;

  double squared_error = 0.0;
  for (const Point& point : samples) {
    const Vector delta = point - centroid;
    const double error = delta * normal;
    squared_error += error * error;
  }
  const double rmse = std::sqrt(squared_error / samples.size());
  const double z = std::abs(normal.z());
  const double angle = axis_angle * CGAL_PI / 180.0;
  const std::uint8_t type = z >= std::cos(angle) ? 1 :
                            z <= std::sin(angle) ? 2 : 0;
  return {normal, centroid, rmse, type};
}

struct Voxel_key {
  std::int64_t x, y, z;
  bool operator==(const Voxel_key& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
  bool operator<(const Voxel_key& other) const {
    if (x != other.x) return x < other.x;
    if (y != other.y) return y < other.y;
    return z < other.z;
  }
};

struct Voxel_hash {
  std::size_t operator()(const Voxel_key& key) const {
    std::size_t seed = std::hash<std::int64_t>{}(key.x);
    seed ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

struct Voxel_data {
  double x = 0, y = 0, z = 0;
  double nx = 0, ny = 0, nz = 0;
  std::size_t count = 0, plane_count = 0;
  Point source;
  double source_distance2 = std::numeric_limits<double>::infinity();
};

void write_voxel_anchors(
    const std::filesystem::path& path, const Point_set& points,
    const Normal_map normals,
    const Point_set::Property_map<std::uint8_t> is_plane,
    const Options& options) {
  std::unordered_map<Voxel_key, Voxel_data, Voxel_hash> voxels;
  voxels.reserve(points.size() / 4);
  const double size = options.anchor_voxel_size;
  for (const Point_set::Index id : points) {
    const Point point = points.point(id);
    const Voxel_key key{
        static_cast<std::int64_t>(std::floor(point.x() / size)),
        static_cast<std::int64_t>(std::floor(point.y() / size)),
        static_cast<std::int64_t>(std::floor(point.z() / size))};
    Voxel_data& voxel = voxels[key];
    voxel.x += point.x(); voxel.y += point.y(); voxel.z += point.z();
    ++voxel.count;
    if (is_plane[id]) {
      const Vector normal = normals[id];
      voxel.nx += normal.x(); voxel.ny += normal.y(); voxel.nz += normal.z();
      ++voxel.plane_count;
    }
  }

  for (auto& [key, voxel] : voxels) {
    voxel.x /= voxel.count; voxel.y /= voxel.count; voxel.z /= voxel.count;
  }
  for (const Point_set::Index id : points) {
    const Point point = points.point(id);
    const Voxel_key key{
        static_cast<std::int64_t>(std::floor(point.x() / size)),
        static_cast<std::int64_t>(std::floor(point.y() / size)),
        static_cast<std::int64_t>(std::floor(point.z() / size))};
    Voxel_data& voxel = voxels.at(key);
    const double dx = point.x() - voxel.x;
    const double dy = point.y() - voxel.y;
    const double dz = point.z() - voxel.z;
    const double distance2 = dx * dx + dy * dy + dz * dz;
    if (distance2 < voxel.source_distance2) {
      voxel.source = point;
      voxel.source_distance2 = distance2;
    }
  }

  std::vector<Voxel_key> keys;
  keys.reserve(voxels.size());
  for (const auto& item : voxels) keys.push_back(item.first);
  std::sort(keys.begin(), keys.end());

  Point_set anchors;
  const Normal_map anchor_normals = anchors.add_normal_map().first;
  const auto source_x = anchors.add_property_map<double>("source_x", 0).first;
  const auto source_y = anchors.add_property_map<double>("source_y", 0).first;
  const auto source_z = anchors.add_property_map<double>("source_z", 0).first;
  const auto anchor_plane = anchors.add_property_map<std::uint8_t>("is_plane", 0).first;
  const auto anchor_type = anchors.add_property_map<std::uint8_t>("plane_type", 0).first;
  const auto plane_ratio = anchors.add_property_map<float>("plane_ratio", 0).first;
  const auto coherence_map = anchors.add_property_map<float>("normal_coherence", 0).first;
  const auto point_count = anchors.add_property_map<std::int32_t>("point_count", 0).first;
  const auto voxel_size = anchors.add_property_map<double>("voxel_size", size).first;
  const auto red = anchors.add_property_map<std::uint8_t>("red", 40).first;
  const auto green = anchors.add_property_map<std::uint8_t>("green", 40).first;
  const auto blue = anchors.add_property_map<std::uint8_t>("blue", 40).first;

  std::size_t planar_count = 0;
  for (const Voxel_key& key : keys) {
    const Voxel_data& voxel = voxels.at(key);
    const double ratio = static_cast<double>(voxel.plane_count) / voxel.count;
    const double normal_length = std::sqrt(
        voxel.nx * voxel.nx + voxel.ny * voxel.ny + voxel.nz * voxel.nz);
    const double coherence = voxel.plane_count ? normal_length / voxel.plane_count : 0;
    const bool planar = ratio >= options.anchor_min_plane_ratio &&
                        coherence >= options.anchor_min_normal_coherence &&
                        normal_length > 1e-12;
    Vector normal(0, 0, 0);
    std::uint8_t type = 0;
    if (planar) {
      normal = Vector(voxel.nx / normal_length, voxel.ny / normal_length,
                      voxel.nz / normal_length);
      const double z = std::abs(normal.z());
      const double angle = options.axis_angle * CGAL_PI / 180.0;
      type = z >= std::cos(angle) ? 1 : z <= std::sin(angle) ? 2 : 0;
    }
    const bool accepted = planar && type != 0;
    const auto inserted = anchors.insert(Point(voxel.x, voxel.y, voxel.z));
    const Point_set::Index id = *inserted;
    source_x[id] = voxel.source.x();
    source_y[id] = voxel.source.y();
    source_z[id] = voxel.source.z();
    anchor_normals[id] = accepted ? normal : Vector(0, 0, 0);
    anchor_plane[id] = accepted;
    anchor_type[id] = accepted ? type : 0;
    plane_ratio[id] = static_cast<float>(ratio);
    coherence_map[id] = static_cast<float>(coherence);
    point_count[id] = static_cast<std::int32_t>(voxel.count);
    voxel_size[id] = size;
    if (accepted && type == 1) {
      red[id] = 30; green[id] = 100; blue[id] = 255;
    } else if (accepted && std::abs(normal.x()) >= std::abs(normal.y())) {
      red[id] = 255; green[id] = 50; blue[id] = 50;
    } else if (accepted) {
      red[id] = 50; green[id] = 255; blue[id] = 50;
    }
    planar_count += accepted;
  }

  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream out(path, std::ios::binary);
  CGAL::IO::set_binary_mode(out);
  if (!out || !(out << anchors))
    throw std::runtime_error("failed to write: " + path.string());
  std::cout << "Saved " << anchors.size() << " voxel anchors ("
            << planar_count << " planar) to " << path.string() << '\n';
}

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    namespace fs = std::filesystem;
    if (!fs::is_regular_file(options.input))
      throw std::runtime_error("input does not exist: " + options.input);
    if (fs::exists(options.output) && !options.overwrite)
      throw std::runtime_error("output exists; pass --overwrite: " + options.output);
    fs::path anchor_path = options.anchor_output;
    if (anchor_path.empty()) {
      anchor_path = fs::path(options.output);
      const std::string stem = anchor_path.stem().string() + "_anchors";
      anchor_path.replace_filename(stem + anchor_path.extension().string());
    }
    if (fs::exists(anchor_path) && !options.overwrite)
      throw std::runtime_error("anchor output exists; pass --overwrite: " + anchor_path.string());

    Point_set points;
    std::ifstream input(options.input, std::ios::binary);
    if (!input || !(input >> points) || points.empty())
      throw std::runtime_error("failed to read point cloud: " + options.input);
    std::cout << "Loaded " << points.size() << " points\n";

    if (options.voxel_size > 0) {
      const auto first = CGAL::grid_simplify_point_set(
          points, options.voxel_size,
          CGAL::parameters::point_map(points.point_map()));
      points.remove(first, points.end());
      points.collect_garbage();
      std::cout << "Downsampled to " << points.size() << " points\n";
    }
    if (points.size() < std::max({options.normal_k, options.region_k,
                                  options.min_points}))
      throw std::runtime_error("too few points after downsampling");

    const Normal_map normals = points.add_normal_map().first;
    CGAL::pca_estimate_normals<CGAL::Parallel_if_available_tag>(
        points, options.normal_k,
        CGAL::parameters::point_map(points.point_map()).normal_map(normals));

    Neighbor_query neighbors(points, options.region_k, points.point_map());
    Region_type region_type(points, options.distance, options.normal_angle,
                            options.min_points, points.point_map(), normals);
    Sorting sorting(points, neighbors, points.point_map());
    sorting.sort();
    Region_growing region_growing(
        points, neighbors, region_type, sorting.seed_map());
    std::vector<Region> regions;
    region_growing.detect(std::back_inserter(regions));
    std::cout << "CGAL proposed " << regions.size() << " regions\n";

    Point_set output;
    const Normal_map output_normals = output.add_normal_map().first;
    const auto is_plane = output.add_property_map<std::uint8_t>("is_plane", 0).first;
    const auto plane_type = output.add_property_map<std::uint8_t>("plane_type", 0).first;
    const auto plane_id = output.add_property_map<std::int32_t>("plane_id", -1).first;
    const auto plane_error = output.add_property_map<float>("plane_error", 0.f).first;
    const auto red = output.add_property_map<std::uint8_t>("red", 40).first;
    const auto green = output.add_property_map<std::uint8_t>("green", 40).first;
    const auto blue = output.add_property_map<std::uint8_t>("blue", 40).first;

    std::vector<Point_set::Index> output_ids;
    output_ids.reserve(points.size());
    for (const Point_set::Index input_id : points) {
      const auto inserted = output.insert(points.point(input_id));
      output_ids.push_back(*inserted);
    }

    fs::path summary_path(options.output);
    summary_path.replace_extension(".planes.csv");
    const fs::path output_parent = fs::path(options.output).parent_path();
    if (!output_parent.empty()) fs::create_directories(output_parent);
    std::ofstream summary(summary_path);
    if (!summary) throw std::runtime_error("failed to write: " + summary_path.string());
    summary << "plane_id,type,num_points,nx,ny,nz,cx,cy,cz,rmse\n";

    std::size_t accepted_points = 0;
    std::int32_t accepted_planes = 0;
    for (const Region& region : regions) {
      const Fitted_plane fitted = fit_plane(points, region, options.axis_angle);
      if (fitted.type == 0 || fitted.rmse > options.max_rmse) continue;
      summary << accepted_planes << ',' << static_cast<int>(fitted.type) << ','
              << region.size() << ',' << fitted.normal.x() << ','
              << fitted.normal.y() << ',' << fitted.normal.z() << ','
              << fitted.centroid.x() << ',' << fitted.centroid.y() << ','
              << fitted.centroid.z() << ',' << fitted.rmse << '\n';

      const double nx = std::abs(fitted.normal.x());
      const double ny = std::abs(fitted.normal.y());
      for (const std::size_t offset : region) {
        const Point_set::Index id = output_ids[offset];
        const Point point = output.point(id);
        const double error = std::abs((point - fitted.centroid) * fitted.normal);
        output_normals[id] = fitted.normal;
        is_plane[id] = 1;
        plane_type[id] = fitted.type;
        plane_id[id] = accepted_planes;
        plane_error[id] = static_cast<float>(error);
        if (fitted.type == 1) {
          red[id] = 30; green[id] = 100; blue[id] = 255;
        } else if (nx >= ny) {
          red[id] = 255; green[id] = 50; blue[id] = 50;
        } else {
          red[id] = 50; green[id] = 255; blue[id] = 50;
        }
      }
      accepted_points += region.size();
      ++accepted_planes;
    }

    std::ofstream out(options.output, std::ios::binary);
    CGAL::IO::set_binary_mode(out);
    if (!out || !(out << output))
      throw std::runtime_error("failed to write: " + options.output);
    write_voxel_anchors(anchor_path, output, output_normals, is_plane, options);
    std::cout << "Accepted " << accepted_planes << " ground/wall planes, "
              << accepted_points << '/' << output.size() << " points\n"
              << "Saved " << options.output << "\n"
              << "Saved " << summary_path.string() << "\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
