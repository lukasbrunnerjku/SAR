import bpy
from mathutils import Euler, Vector
import numpy as np
import argparse
import json
import math
from blendtorch import btb

SCN = bpy.context.scene
LAYER = bpy.context.view_layer

cfg = {
    'CameraSettings': {
        'HxW': (640, 512),
        'Lane': ((-30, 0, 30), (30, 0, 30)),  # start, stop in m
        'Npos': 12,  # images per lane
        'Randomize': True,
        'StdDev': 1,  # perturbate positions in m
        'Lens': (48, 52),  # focal length in mm
    },
    'ParticleSettings014': {
        'Number': (20, 30),
        'Scale': (0.6, 0.7),
        'Randomize Phase': (0.8, 1.3),
        'Scale Randomness': (0, 0.15),
    },
    'ParticleSettings23': {
        'Number': (10, 20),
        'Scale': (0.6, 0.7),
        'Randomize Phase': (0, 0.1),
        'Scale Randomness': (0, 0.15),
    },
    'TreeSettings': {
        'N': 20,
        'Seed': (0, 1234),
        'Fac': (0.7, 0.9),
        'Bright': (0.2, 1.0),
        'Scale': (0.9, 1.1),
        'Class Probabilities': None,
        # angle: z-axis rotation in radians
        'Angle': (0, np.pi),  
        # area: x_min, y_min, x_max, y_max in m
        'Area': (-40, -25, 40, 25),  
    },
    'HumanSettings': {
        'N': 5,
        'Scale': (0.9, 1.1),
        'Class Probabilities': None,
        'Strength': (0.5, 0.8), 
        # angle: z-axis rotation in radians
        'Angle': (0, 2 * np.pi),  
        # area: x_min, y_min, x_max, y_max in m
        'Area': (-30, -15, 30, 15), 
    },
    'GroundSettings': {
        'Scale': (0.22, 0.42),
    }, 
}

ps014 = cfg['ParticleSettings014']
ps23 = cfg['ParticleSettings23']
ts = cfg['TreeSettings']
hs = cfg['HumanSettings']
cs = cfg['CameraSettings']
gs = cfg['GroundSettings']

tcoll = SCN.collection.children['Trees']
bcoll = SCN.collection.children['Branches']
hcoll = SCN.collection.children['Humans']  
gcoll = SCN.collection.children['Generated']  

def parse_additional_args(remainder):
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-config')
    return parser.parse_args(remainder)

def rotate_bone(pose_name, bone_name, axis=None, angle=None, xyz_offset=None):
    pose_obj = bpy.data.objects[pose_name]  # get pose object
    LAYER.objects.active = pose_obj  # select pose object
    bpy.ops.object.mode_set(mode='POSE')  # go to pose mode
    bone = pose_obj.pose.bones[bone_name]  # get specified bone
    if xyz_offset is not None:  # rel. rotation (x, y, z) in radians
        bone.rotation_euler.rotate(Euler(xyz_offset))
    elif axis is not None and angle is not None:
        bone.rotation_euler.rotate_axis(axis, math.radians(angle))
    bpy.ops.object.mode_set(mode='OBJECT')  # set back to object mode

def get_bboxes(cam, humans):
    bboxes = []
    for human in humans:
        # 2D points in cam space of 3D bbox => 8 x 2 are (x, y)
        xy = cam.bbox_object_to_pixel(human)
        h, w = cs['HxW']
        xy[:, 0] = np.clip(xy[:, 0], 0, w-1)  # bound x by [0, w-1]
        xy[:, 1] = np.clip(xy[:, 1], 0, h-1)  # bound y by [0, h-1]

        # 2D bbox format: xmin, ymin, xmax, ymax
        bbox = (min(xy[:, 0]), min(xy[:, 1]), max(xy[:, 0]), 
            max(xy[:, 1]))

        # xmin = xmax or ymin = ymax => not inside camera view
        if bbox[0] == bbox[2] or bbox[1] == bbox[3]:  
            continue
        bboxes.append(bbox)  # 4,

    if len(bboxes) >= 1:
        return np.array(bboxes)  # nhuman x 4
    else:  # invalid bboxes
        return np.array([])  # shape: 0,

def remove_objects():
    for obj in gcoll.objects: 
        if isinstance(obj.data, bpy.types.Mesh):
            if len(obj.data.materials) != 0:
                obj.data.materials.pop(index=0) 
                                 
        bpy.data.objects.remove(obj, do_unlink=True)
    
    for m in list(bpy.data.materials):
        if m.users == 0:  # remove unused materials
            bpy.data.materials.remove(m, do_unlink=True)     

    for m in list(bpy.data.meshes):
        if m.users == 0:  # remove unused meshes
            bpy.data.meshes.remove(m, do_unlink=True) 
    
    for m in list(bpy.data.armatures):
        if m.users == 0:  # remove unused armatures
            bpy.data.armatures.remove(m, do_unlink=True)

def cam_positions(cam):
    for pos in np.linspace(*cs['Lane'], num=cs['Npos']):
        if cs['Randomize']:
            std = cs['StdDev']  # perturbate with gaussian noise
            look_at = pos + np.random.normal(scale=std, size=(3,))
            look_at[2] = 0  # z=0, look at ground plane
            look_from = pos + np.random.normal(scale=std, size=(3,))
        else:
            look_at = pos.copy()
            look_at[2] = 0  # z=0, look at ground plane
            look_from = pos

        cam.look_at(look_at, look_from)
        yield look_at, look_from

def random_placement(obj):
    # randomly place in xy plane inside specified area
    if 'pose' in obj.name:  # human 
        area, angle = hs['Area'], hs['Angle']
    elif 'tree' in obj.name:  # tree
        area, angle = ts['Area'], ts['Angle']
    else:
        raise AttributeError

    xy = np.random.uniform(area[:2], area[2:], size=2)
    obj.location.xy = xy  

    # random rel. rotation around z-axis for objects
    angle = np.random.uniform(*angle)  # in radians
    obj.rotation_euler.z += angle  

def randomize_branch01234_material(nr: int = 0):
    # branch object for particle settings
    plane = bcoll.objects[f'branch{nr}']
    # manipulate branch textures from this plane's material
    mat = plane.data.materials[0]
    # randomize material properties
    nodes = mat.node_tree.nodes
    mix = nodes.get('Mix Shader')
    bright = nodes.get('Bright/Contrast')
    # control branch transparency
    mix.inputs['Fac'].default_value = np.random.uniform(*ts['Fac'])
    # control branch emission brigthness 
    bright.inputs['Bright'].default_value = np.random.uniform(*ts['Bright']) 
    
def randomize_tree014_particle_system(nr: int = 0):
    mod = tcoll.objects[f'tree{nr}'].modifiers['ParticleSettings']
    settings = mod.particle_system.settings
    
    settings.instance_object = bcoll.objects[f'branch{nr}']
    settings.count = np.random.randint(*ps014['Number'])
    settings.particle_size = np.random.uniform(*ps014['Scale'])
    settings.phase_factor_random = np.random.uniform(*ps014['Randomize Phase'])
    settings.size_random = np.random.uniform(*ps014['Scale Randomness'])

def randomize_tree23_particle_system(nr: int = 2):
    mod = tcoll.objects[f'tree{nr}'].modifiers['ParticleSettings']
    settings = mod.particle_system.settings
    
    settings.instance_object = bcoll.objects[f'branch{nr}']
    settings.count = np.random.randint(*ps23['Number'])
    settings.particle_size = np.random.uniform(*ps23['Scale'])
    settings.phase_factor_random = np.random.uniform(*ps23['Randomize Phase'])
    settings.size_random = np.random.uniform(*ps23['Scale Randomness'])

def create_tree(): 
    # randomly create one type of tree object
    tree_objs = [obj for obj in tcoll.objects if 'tree' in obj.name]

    ids = np.arange(len(tree_objs))
    if ts['Class Probabilities'] is None:
        p = np.ones(len(tree_objs))
    else:
        p = np.array(ts['Class Probabilities'])
    p /= p.sum()  # normalize
    nr = np.random.choice(ids, p=p)  # choose one
    
    # tree with particle modifier for branches
    tree = tcoll.objects[f'tree{nr}'].copy()
    mod = tree.modifiers['ParticleSettings']
    system = mod.particle_system
    
    # seed is not shared for tree objects
    system.seed = np.random.randint(*ts['Seed'])
    
    # random rescale (particles excluded)
    tree.scale *= np.random.uniform(*ts['Scale'])
    
    random_placement(tree)
    
    gcoll.objects.link(tree)
    return tree

def create_human():
    pose_objs = [obj for obj in hcoll.objects if 'pose' in obj.name]
    mesh_objs = [obj for obj in hcoll.objects if 'male' in obj.name]

    ids = np.arange(len(mesh_objs))
    if hs['Class Probabilities'] is None:
        p = np.ones(len(mesh_objs))
    else:
        p = np.array(hs['Class Probabilities'])
    p /= p.sum()  # normalize
    nr = np.random.choice(ids, p=p)  # choose one

    pose_obj = pose_objs[nr].copy()
    mesh_obj = mesh_objs[nr].copy()
    matrixcopy = mesh_obj.matrix_world.copy()

    mat = mesh_objs[nr].data.materials[0].copy()  # copy material
    mesh_obj.data = mesh_obj.data.copy()  # copy mesh
    
    # randomize material
    nodes = mat.node_tree.nodes
    emission = nodes.get('Emission')
    emission.inputs['Strength'].default_value = np.random.uniform(*hs['Strength']) 

    # re-assign randomized material
    mesh_obj.data.materials[0] = mat

    mesh_obj.parent = None
    mesh_obj.modifiers['Armature'].object = None
    mesh_obj.parent = pose_obj
    # keep transformation from before when child was linked
    # to the parent, otherwise both objects will be at different
    # locations and applying the armature will rip mesh appart 
    mesh_obj.matrix_world = matrixcopy
    mesh_obj.modifiers['Armature'].object = pose_obj
    
    pose_obj.scale *= np.random.uniform(*hs['Scale'])
    
    # object will follow pose when moved with a armature modifier
    random_placement(pose_obj)
    
    # make visible by linking
    gcoll.objects.link(pose_obj)
    gcoll.objects.link(mesh_obj)
    return pose_obj
        
def forcefield2D(objs, alpha=+10, beta=1, rmin=1, rmax=10):    
        # pos. alpha = repulsion, neg. = attraction
        displacements = []  
        for j in range(len(objs)):
            force = Vector([0, 0])  # total force on obj j
            pj = objs[j].location.xy  # Vector
            for i in range(len(objs)):
                if i != j:
                    pi = objs[i].location.xy  # Vector
                    r = (pj - pi).magnitude
                    if r >= rmax:
                        continue
                    dir = (pj - pi) / r  # direction to pj
                    r = max(r, rmin)  # avoid too high force
                    force += alpha / (r ** beta) * dir  
            
            displacements.append(force)  # total force
            
        # update all obj positions simultaniously
        for obj, dis in zip(objs, displacements):
            obj.location.xy += dis

def randomize_scene():
    # once per scene (lane) set different material settings
    # for the branch materials used in trees' particle systems,
    # this systems are shared for the same type of tree, except 
    # the seed and assign different settings for the particle systems
    for nr in (0, 1, 2, 3, 4):
        randomize_branch01234_material(nr)
    for nr in (0, 1, 4):
        randomize_tree014_particle_system(nr)
    for nr in (2, 3):
        randomize_tree23_particle_system(nr)

    mat = bpy.data.materials['ground']  # randomize ground material
    mapping = mat.node_tree.nodes.get('Mapping')
    scale = np.ones((3,))
    scale[:2] = np.random.uniform(*gs['Scale'], size=2)
    mapping.inputs['Scale'].default_value = scale
    
    if cs['Randomize']:  # randomize camera
        camera = bpy.data.cameras['Camera']
        camera.lens =  np.random.uniform(*cs['Lens'])

def create_scene():
    randomize_scene()

    # create new randomized objects from templates
    humans = [create_human() for _ in range(hs['N'])]
    trees = [create_tree() for _ in range(ts['N'])]
    
    return humans, trees

def main():
    btargs, _ = btb.parse_blendtorch_args()

    humans, trees = None, None
    camera_genenerator = None

    build_scene = create_scene

    def recreate_scene():  
        nonlocal humans, trees  # reuse scene objects
        randomize_scene()

        for pose_obj in humans:
            pose_obj.scale *= np.random.uniform(*hs['Scale'])

            mesh_obj = ?
            mat = mesh_obj.data.materials[0]
            nodes = mat.node_tree.nodes
            emission = nodes.get('Emission')
            emission.inputs['Strength'].default_value = np.random.uniform(*hs['Strength'])
            mesh_obj.data.materials[0] = mat

            random_placement(pose_obj)

        # randomize
        for tree in trees:
            mod = tree.modifiers['ParticleSettings']
            system = mod.particle_system
            # seed is not shared for tree objects
            system.seed = np.random.randint(*ts['Seed'])

            # random rescale (particles excluded)
            tree.scale *= np.random.uniform(*ts['Scale'])

            random_placement(tree)

    def pre_animation(cam):
        nonlocal humans, trees, camera_genenerator, build_scene      
        humans, trees = build_scene()
        build_scene = recreate_scene

        # move close by objects to less dense regions
        forcefield2D(humans + trees)

        camera_genenerator = cam_positions(cam)

    def pre_frame():
        # set camera to initial position
        next(camera_genenerator)  

    def post_frame(off, pub, cam):
        bboxes = get_bboxes(cam, humans)
        
        pub.publish(
            image=off.render(),  # h x w x 3
            # note: 
            bboxes=bboxes,  # nhuman x 4; nhuman can be 0!
            cids=np.ones((len(bboxes), )),  # class labels
        )

    def post_animation():
        nonlocal humans, trees, camera_genenerator 
        remove_objects()
        humans, trees = None, None
        camera_genenerator = None
        
    # so every Blender has its own random seed
    np.random.seed(btargs.btseed)

    # data channel to connect to pytorch
    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid)

    # setup default image rendering
    cam = btb.Camera(shape=cs['HxW'])
    off = btb.OffScreenRenderer(camera=cam, mode='rgb', gamma_coeff=2.2)
    off.set_render_style(shading='RENDERED', overlays=False)

    # setup the animation and run endlessly
    anim = btb.AnimationController()
    # invoked before first frame of animation range is processed
    anim.pre_animation.add(pre_animation, cam)
    # invoked before a frame begins
    anim.pre_frame.add(pre_frame)
    # invoked after a frame is finished
    anim.post_frame.add(post_frame, off, pub, cam)
    # invoked after the last animation frame has completed
    anim.post_animation.add(post_animation)

    # start the animation loop
    anim.play(frame_range=(0, cs['Npos']-1), num_episodes=-1, 
        use_physics=False, use_animation=False)

main()
