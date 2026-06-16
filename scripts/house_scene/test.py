import os
import json
import genesis as gs

# 1. Configuration Setup
JSON_CONFIG_PATH = "./data/replica_cad/configs/scenes/apt_0.scene_instance.json"
ASSET_ROOT_DIR = os.path.abspath("./data/replica_cad/")

gs.init(backend=gs.cpu)
scene = gs.Scene(
    show_viewer=True,
    
    # 1. Slow down time (The Timestep)
    sim_options=gs.options.SimOptions(
        dt=0.005,  # Default is usually 0.01 or higher. A smaller dt (e.g., 200Hz) 
                   # takes smaller steps in time, allowing the engine to gently ease 
                   # overlapping objects apart before the forces multiply to infinity.
    ),
    
    rigid_options=gs.options.RigidOptions(
        multiplier_collision_broad_phase=50,
        max_collision_pairs=10000,
        iterations=100, # Replaces the hallucinated 'solver_iterations'
    )
)

def convert_habitat_to_genesis(pos, quat_wxyz):
    """
    Swizzles Habitat's Y-up coordinates to Genesis's Z-up environment.
    """
    x, y, z = pos
    qw, qx, qy, qz = quat_wxyz
    
    new_pos = [x, -z, y]
    new_quat_wxyz = [qw, qx, -qz, qy]
    
    return new_pos, new_quat_wxyz

def resolve_asset_path(template_path, root_dir):
    """
    Strips directory prefixes (like 'objects/' or 'stages/') from the JSON key
    and hunts for the matching Habitat configuration file.
    """
    # Extract just the base name (e.g., "objects/frl_apartment_basket" -> "frl_apartment_basket")
    base_name = os.path.basename(template_path)
    
    search_dir = os.path.join(root_dir, "configs")
    if not os.path.exists(search_dir):
        search_dir = root_dir 

    for subdir, dirs, files in os.walk(search_dir):
        for file in files:
            if file.startswith(base_name) and file.endswith(".json"):
                json_path = os.path.join(subdir, file)
                try:
                    with open(json_path, "r") as f:
                        obj_config = json.load(f)
                    
                    # Grab the relative pointer
                    asset_rel_path = obj_config.get("urdf_filepath") or obj_config.get("render_asset")
                    if asset_rel_path:
                        abs_path = os.path.abspath(os.path.join(os.path.dirname(json_path), asset_rel_path))
                        if os.path.exists(abs_path):
                            return abs_path
                except Exception:
                    pass
    return None

def spawn_entity(template_name, asset_path, pos, quat, is_fixed, scale=1.0):
    try:
        if asset_path.endswith(".urdf"):
            scene.add_entity(gs.morphs.URDF(file=asset_path, pos=pos, quat=quat, fixed=is_fixed))
        elif asset_path.endswith(".glb") or asset_path.endswith(".obj"):
            scene.add_entity(gs.morphs.Mesh(file=asset_path, pos=pos, quat=quat, fixed=is_fixed, scale=(scale, scale, scale)))
        print(f"✅ Spawned: {os.path.basename(template_name)}")
    except Exception as e:
        print(f"❌ Failed to load {os.path.basename(template_name)}: {e}")

# Read your exact JSON payload
with open(JSON_CONFIG_PATH, "r") as f:
    config = json.load(f)

# 3. Parse Room Shell Background (Using your specific stage name)
print("\n--- Processing Stage (Room Shell) ---")
stage_info = config.get("stage_instance", {})
stage_template = stage_info.get("template_name")  # "stages/frl_apartment_stage"

if stage_template:
    room_asset = resolve_asset_path(stage_template, ASSET_ROOT_DIR)
    if room_asset:
        room_pos, room_quat = convert_habitat_to_genesis(
            stage_info.get("translation", [0, 0, 0]), 
            stage_info.get("rotation", [1, 0, 0, 0]) 
        )
        # Force fixed=True so the apartment doesn't fall out of the sky
        spawn_entity(stage_template, room_asset, pos=room_pos, quat=room_quat, is_fixed=True)
    else:
        print(f"⚠️ Could not resolve stage asset for {stage_template}.")

# 4. Parse Rigid Objects
print("\n--- Processing Rigid Objects ---")
for obj in config.get("object_instances", []):
    template_name = obj["template_name"]
    asset_path = resolve_asset_path(template_name, ASSET_ROOT_DIR)
    
    if asset_path:
        new_pos, new_quat = convert_habitat_to_genesis(obj["translation"], obj["rotation"])
        
        # In Habitat, STATIC motion type implies fixed geometry
        is_fixed = obj.get("motion_type") == "STATIC"
        
        spawn_entity(
            template_name=template_name,
            asset_path=asset_path,
            pos=new_pos,
            quat=new_quat,
            is_fixed=is_fixed,
            scale=obj.get("uniform_scale", 1.0)
        )

# 5. Parse Articulated Objects
print("\n--- Processing Articulated Objects ---")
for obj in config.get("articulated_object_instances", []):
    template_name = obj["template_name"]
    asset_path = resolve_asset_path(template_name, ASSET_ROOT_DIR)
    
    if asset_path:
        new_pos, new_quat = convert_habitat_to_genesis(obj["translation"], obj["rotation"])
        spawn_entity(
            template_name=template_name,
            asset_path=asset_path,
            pos=new_pos,
            quat=new_quat,
            is_fixed=obj.get("fixed_base", True),
            scale=obj.get("uniform_scale", 1.0)
        )

# 6. Compile and Run
print("\nCompiling environment stage...")
scene.build()

while True:
    scene.step()
