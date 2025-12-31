import bpy
import math
import os
import sys
import argparse
import json
import mathutils

# Zero123++ v1.2 Constants
ELEVATIONS = [20.0, -10.0, 20.0, -10.0, 20.0, -10.0]
AZIMUTHS = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]
FOV = 30.0  # Degrees
RESOLUTION = 320
DISTANCE = 1.5

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_camera():
    cam_data = bpy.data.cameras.new('Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # Set FOV
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians(FOV)
    
    return cam_obj

def setup_lighting():
    # Add an area light
    light_data = bpy.data.lights.new(name="Light", type='AREA')
    light_data.energy = 1000
    light_obj = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (0, 0, 5)
    
    # Add ambient light (world)
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[1].default_value = 1.0  # Strength

def normalize_object(obj):
    """
    Scale and center object to fit in unit sphere.
    """
    # Clear parent inverse to get actual bounds
    if obj.parent:
        mat = obj.matrix_world
        obj.parent = None
        obj.matrix_world = mat

    # Center object
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)
    
    # Scale
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        
    # Compute max dimension
    # Simple approximation using dimensions
    dims = obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    
    if max_dim > 0:
        scale = 1.0 / (max_dim / 1.5) # Scale to fit comfortably
        obj.scale = (scale, scale, scale)
        
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

def look_at(obj, target):
    """
    Rotate camera to look at target.
    """
    direction = target - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def set_camera_pose(cam_obj, elevation_deg, azimuth_deg, distance):
    elev_rad = math.radians(elevation_deg)
    azim_rad = math.radians(azimuth_deg)
    
    x = distance * math.cos(elev_rad) * math.cos(azim_rad)
    y = distance * math.cos(elev_rad) * math.sin(azim_rad)
    z = distance * math.sin(elev_rad)
    
    cam_obj.location = (x, y, z)
    
    # Point camera at origin
    # Blender 'Track To' constraint
    constraint = cam_obj.constraints.new(type='TRACK_TO')
    target = bpy.data.objects.new("Target", None)
    bpy.context.collection.objects.link(target)
    target.location = (0,0,0)
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

# ---- Main Execution ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Blender specific argument handling: all args after "--"
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []

    parser.add_argument("input_path", help="Path to .glb file")
    parser.add_argument("--output_dir", default="rendered_views", help="Output directory")
    args = parser.parse_args(argv)
    
    input_path = args.input_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Scene
    reset_scene()
    cam = setup_camera()
    setup_lighting()
    
    # Import GLB
    bpy.ops.import_scene.gltf(filepath=input_path)
    
    # Get imported object (assume it's the first mesh object found)
    mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
    if not mesh_objs:
        print("No mesh found in imported GLB.")
        sys.exit(1)
        
    # Join all meshes into one if multiple exist
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    
    main_obj = bpy.context.view_layer.objects.active
    normalize_object(main_obj)
    
    # Set Render Settings
    bpy.context.scene.render.resolution_x = RESOLUTION
    bpy.context.scene.render.resolution_y = RESOLUTION
    bpy.context.scene.render.film_transparent = True
    
    # Render Views
    camera_metadata = []
    
    for i, (elev, azim) in enumerate(zip(ELEVATIONS, AZIMUTHS)):
        # Remove old constraints to reset
        for c in cam.constraints:
            cam.constraints.remove(c)
            
        set_camera_pose(cam, elev, azim, DISTANCE)
        
        # Add constraint again
        constraint = cam.constraints.new(type='TRACK_TO')
        # Re-find target or create simplified look-at logic
        # For simplicity, let's just use the track-to again
        target = [o for o in bpy.data.objects if o.name == "Target"][0]
        constraint.target = target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        
        # Render
        filename = f"{i:03d}.png"
        filepath = os.path.join(output_dir, filename)
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        
        # Save Metadata (3x4 Matrix: [R|t])
        # Blender world matrix is 4x4.
        matrix_list = [list(row) for row in cam.matrix_world]
        
        camera_metadata.append({
            "view_idx": i,
            "elevation": elev,
            "azimuth": azim,
            "matrix_world": matrix_list
        })
        
    # Save Metadata JSON
    with open(os.path.join(output_dir, "transforms.json"), 'w') as f:
        json.dump(camera_metadata, f, indent=2)
        
    print(f"Rendered {len(ELEVATIONS)} views to {output_dir}")
