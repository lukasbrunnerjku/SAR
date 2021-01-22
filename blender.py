import bpy
from mathutils import Euler
import numpy as np
import argparse
import json
import sys
import math
from blendtorch import btb

SCN = bpy.context.scene
LAYER = bpy.context.view_layer

DEFAULT_CONFIG = {
    ### camera ###
    'camera.shape': (640, 512),  # height and width of camera image
    'camera.start': (0, 0, 90),
    'camera.stop': (50, 0, 90),
    'camera.npos': 10,
    'camera.randomize': True,
    'camera.scale': 2,  # standard deviation gaussian of position
    'camera.focalrange': (48, 52),  # focal length in mm
    ### human ### 
    'scene.nhuman': 1,
    'scene.humanclassprob': None,
    'scene.humanheatrange': (0.7, 1.0),
    'scene.humanrotation': (0, 2*np.pi),  # around z-axis
    ### tree ###
    'scene.ntree': 1,
    'scene.treeclassprob': (0.7, 0.1, 0.1, 0.1),
    'scene.treeheatrange': (0.6, 0.9),
    'scene.treealpharange': (0.05, 0.1),
    'scene.treerotation': (0, 2*np.pi),  # around z-axis
    ### ground ###
    # 'ground.texture': ?,  # select texture file
    'ground.texturenoise': 0.2,  # pertubate texture mapping
    # 2D bounding box in which objects can live
    'scene.bbox': (-60, -30, 60, 30),  # x_min, y_min, x_max, y_max
}

def parse_additional_args(remainder):
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-config')
    return parser.parse_args(remainder)

def main():
    btargs, remainder = btb.parse_blendtorch_args()
    otherargs = parse_additional_args(remainder)

    if otherargs.json_config is not None:
        print('Got custom configuration file.')
        with open(otherargs.json_config, 'r') as fp:
            cfg = json.loads(fp.read())
    else:
        cfg = DEFAULT_CONFIG               

    humans, trees, cam_gen = None, None, None

    def lane_pos_gen(cam):
        start = cfg['camera.start']
        stop = cfg['camera.stop']
        npos =  cfg['camera.npos']
        randomize = cfg['camera.randomize']
        scale = cfg['camera.scale']
        
        positions = np.linspace(start, stop, num=npos)
        for pos in positions:
            if randomize:
                # perturbate with gaussian noise
                look_at = pos + np.random.normal(scale=scale, size=(3,))
                look_from = pos + np.random.normal(scale=scale, size=(3,))
                look_at[2] = 0  # z=0, look at ground plane
            else:
                look_at = pos.copy()
                look_from = pos
                look_at[2] = 0  # z=0, look at ground plane

            cam.look_at(look_at, look_from)
            yield look_at, look_from

    def randomize_ground_material(mat, cfg):
        nodes = mat.node_tree.nodes
        mapping = nodes.get('Mapping')
        noise = cfg['ground.texturenoise']

        location = np.zeros((3,))  # defaults
        # add perturbations
        location[:2] += np.random.uniform(-noise, +noise, size=2)
        mapping.inputs['Location'].default_value = location

        scale = np.ones((3,))  # defaults
        # add perturbations
        scale[:2] += np.random.uniform(-noise, +noise, size=2)
        mapping.inputs['Scale'].default_value = scale

        return mat

    def randomize_tree_material(mat, cfg):
        nodes = mat.node_tree.nodes
        bsdf = nodes.get('Principled BSDF')
        emission = nodes.get('Emission')

        emission.inputs['Strength'].default_value = np.random.uniform(*cfg['scene.treeheatrange']) 
        bsdf.inputs['Alpha'].default_value = np.random.uniform(*cfg['scene.treealpharange']) 

        return mat

    def randomize_human_material(mat, cfg):
        nodes = mat.node_tree.nodes
        emission = nodes.get('Emission')
        emission.inputs['Strength'].default_value = np.random.uniform(*cfg['scene.humanheatrange']) 

        return mat

    def create_tree(cfg):
        tcoll = SCN.collection.children['Trees']      
        gcoll = SCN.collection.children['Generated']
        
        mesh_objs = list(tcoll.objects)
        ids = np.arange(len(mesh_objs))
        if cfg['scene.treeclassprob'] is None:
            p = np.ones(len(mesh_objs))
        else:
            p = np.array(cfg['scene.treeclassprob'])
        p /= p.sum()  # make sure probs are normalized
        c = np.random.choice(ids, p=p)  # choose 1 id from ids
        
        # make copy to not alter the original template
        # note: linked material is not copied implicitly
        mesh_obj = mesh_objs[c].copy()
        mat = mesh_obj.data.materials[0].copy()  # copy material
        mesh_obj.data = mesh_obj.data.copy()  # copy mesh
        # assign the randomized copied material to the mesh object
        mesh_obj.data.materials[0] = randomize_tree_material(mat, cfg)
        
        # make visible under collection 'Generated'
        gcoll.objects.link(mesh_obj)  # link mesh Object (Mesh)
        
        xy = np.random.uniform(cfg['scene.bbox'][:2], 
            cfg['scene.bbox'][2:], size=2)
        mesh_obj.location.xy = xy  # randomly place in xy plane
        
        angle = np.random.uniform(cfg['scene.treerotation'][0], 
            cfg['scene.treerotation'][1])  # in radians
        mesh_obj.rotation_euler.z += angle  # random rel. rotation around z-axis

        return mesh_obj
    
    def create_human(cfg):  # SCN = bpy.context.scene
        tcoll = SCN.collection.children['Humans']      
        gcoll = SCN.collection.children['Generated']

        pose_objs = [obj for obj in tcoll.objects if 'pose' in obj.name]
        mesh_objs = [obj for obj in tcoll.objects if 'male' in obj.name]

        ids = np.arange(len(mesh_objs))
        if cfg['scene.humanclassprob'] is None:
            p = np.ones(len(mesh_objs))
        else:
            p = np.array(cfg['scene.humanclassprob'])
        p /= p.sum()  # make sure probs are normalized
        c = np.random.choice(ids, p=p)  # choose 1 id from ids

        # make copy to not alter the original template
        # note: nothing is copied implicitly in Blender for 
        # performance reasons, thus we have to do it manually
        pose_obj = pose_objs[c].copy()
        mesh_obj = mesh_objs[c].copy()
        mat = mesh_obj.data.materials[0].copy()  # copy material
        mesh_obj.data = mesh_obj.data.copy()  # copy mesh
        # assign the randomized copied material to the mesh object
        mesh_obj.data.materials[0] = randomize_human_material(mat, cfg)
        
        #print(mesh_obj.data, mesh_obj.active_material)
        #print(mesh_obj.find_armature())
        #mesh_obj.modifiers.clear()
        mesh_obj.parent = None
        
        # TODO: link the mesh_obj to the pose_obj...
        # when linking the mesh to the pose it gets all messed up
        # I think its a parent inverse problem because when 
        # linking the mesh moves up in the air and is upside down vs.
        # with parent = None mesh stays where it should
        
        #mesh_obj.parent = pose_obj  # set pose to be parent of mesh 
        # neutralize parent inverse from previous parenting
        #mesh_obj.matrix_basis = mesh_obj.matrix_parent_inverse @ mesh_obj.matrix_basis
        #mesh_obj.matrix_parent_inverse.identity()
        
        # modifiers: Armature will be linked to the original pose
        # we need to link it with the copied pose object 
        # to be deformed accordingly
        mesh_obj.modifiers['Armature'].object = pose_obj
        
        # note: to work properly the center of the pose and mesh must be equal,
        # centers can be set in Object Mode under:
        # Object > set Origin > Origin to Center of Mass (Volume)
        
        # make visible under collection 'Generated'
        gcoll.objects.link(pose_obj)  # link pose Object (Armature)
        gcoll.objects.link(mesh_obj)  # link mesh Object (Mesh)
        
        # get active object:
        #obj = bpy.context.object
        # different ways of location and rotation displacements:
        #obj.rotation_euler.x += x_offset  # in radians
        #obj.rotation_euler.rotate(Euler((x_offset, y_offset, z_offset)))  # in radians
        #obj.rotation_euler = (x_value, y_value, z_value)
        #obj.rotation_euler.rotate_axis('X', math.radians(45))
        # note: avoid bpy.ops since operators do a scene update
        # before continuing, which is expensive
        
        # Object will follow Pose when moved with a Armature Modifier applied
        xy = np.random.uniform(cfg['scene.bbox'][:2], 
            cfg['scene.bbox'][2:], size=2)
        pose_obj.location.xy = xy  # randomly place in xy plane
        mesh_obj.location.xy = xy
        
        angle = np.random.uniform(cfg['scene.humanrotation'][0], 
            cfg['scene.humanrotation'][1])  # in radians
        pose_obj.rotation_euler.z += angle  # random rel. rotation around z-axis
        mesh_obj.rotation_euler.z += angle 
        
        return mesh_obj
        
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

    def randomize_cam_params(cam_name, cfg):
        camera = bpy.data.cameras[cam_name]  # get the specified camera
        camera.lens =  np.random.uniform(*cfg['camera.focalrange'])

    def remove_objects():
        gcoll = SCN.collection.children['Generated']

        for obj in gcoll.objects: 
            if isinstance(obj.data, bpy.types.Mesh):
                # bpy.types.Armature has no attribute 'materials'
                if len(obj.data.materials) != 0:
                    obj.data.materials.pop(index=0) 

            # here we remove bpy.types.Mesh and bpy.types.Armature objects
            # note: after removal objects will still be accessable via
            # bpy.data.meshes and bpy.data.armatures!                     
            bpy.data.objects.remove(obj, do_unlink=True)
            
        # before we delete unused materials set 'ground' material 
        # back to the original one or the original material is lost:
        ground = bpy.data.objects.get('ground')
        ground.data.materials[0] = bpy.data.materials['ground_heat']
        
        for m in list(bpy.data.materials):
            if m.users == 0:  # remove unused materials
                bpy.data.materials.remove(m, do_unlink=True)     

        for m in list(bpy.data.meshes):
            if m.users == 0:  # remove unused meshes
                bpy.data.meshes.remove(m, do_unlink=True) 
        
        for m in list(bpy.data.armatures):
            if m.users == 0:  # remove unused armatures
                bpy.data.armatures.remove(m, do_unlink=True)

    def create_scene(cfg):    
        # create humans and trees with randomized materials 
        humans = [create_human(cfg) for _ in range(cfg['scene.nhuman'])]
        trees = [create_tree(cfg) for _ in range(cfg['scene.ntree'])]
            
        # randomize ground material
        ground = bpy.data.objects.get('ground')
        mat = bpy.data.materials['ground_heat'].copy()
        ground.data.materials[0] = randomize_ground_material(mat, cfg)

        # randomize camera
        if cfg['camera.randomize']:
            randomize_cam_params('Camera', cfg)
        
        return humans, trees

    def get_bboxes(cam, humans):
        bboxes = []
        for human in humans:
            # 2D points in cam space of 3D bbox
            xy = cam.bbox_object_to_pixel(human)  # 8 x 2 are (x, y)
            # calculate 2D bbox in format: xmin, ymin, xmax, ymax
            bbox = (min(xy[:, 0]), min(xy[:, 1]), max(xy[:, 0]), 
                max(xy[:, 1]))
            bboxes.append(bbox)  # 4,
        return np.stack(bboxes, axis=0)  # nhuman x 4

    def pre_animation(cam):
        nonlocal humans, trees, cam_gen      
        humans, trees = create_scene(cfg)
        cam_gen = lane_pos_gen(cam)  # setup cam position generator

    def pre_frame():
        next(cam_gen)  # set camera to initial position

    def post_frame(off, pub, cam):
        bboxes = get_bboxes(cam, humans)
        
        pub.publish(
            image=off.render(),  # h x w x 3
            bboxes=bboxes,  # n x 4
            cids=np.zeros((len(bboxes), )),
        )

    def post_animation():
        nonlocal humans, trees, cam_gen
        remove_objects()
        humans, trees, cam_gen = None, None, None
        
    # make sure every Blender has its own random seed
    np.random.seed(btargs.btseed)

    # data source
    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid)

    # setup default image rendering
    cam = btb.Camera(shape=cfg['camera.shape'])
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
    """
    num_episodes: int
        The number of loops to play. -1 loops forever.
    use_animation: bool
        Whether to use Blender's non-blocking animation system or use a 
        blocking variant. By default True. When True, allows BlenderUI 
        to refresh and be responsive. The animation will be run in 
        target FPS. When false, does not allow Blender UI to refresh. 
        The animation runs as fast as it can.
    """
    npos = cfg['camera.npos']  # each frame has a different cam position
    # start the animation loop
    anim.play(frame_range=(0, npos), num_episodes=-1, use_physics=False, 
        use_animation=False)
    # note: with use_animation we can watch the camera changing position
    # inside Blender, to actually see the animation we have to switch
    # to the animation view before starting this script from the pytorch side
    # via >> python pytorch.py 

main()
