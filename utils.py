import torch

def data_mean_and_std(dataloader):
    # dataloader should yield non-normalized images with 
    # floating point values in [0, 1] range
    # note: Var[x] = E[X^2] - E^2[X]
    N = 0
    C = next(iter(dataloader)).size(1)
    channelwise_sum = torch.zeros(C)
    channelwise_sum_squared = torch.zeros(C)
    for images in dataloader:
        N += images.size(0) * images.size(2) * images.size(3) # pixels per channel
        channelwise_sum += images.sum([0, 2, 3])  # C,
        channelwise_sum_squared += torch.square(images).sum([0, 2, 3])  # C,
    
    mean = channelwise_sum / N  # C,
    std = torch.sqrt(channelwise_sum_squared / N - torch.square(mean))  # C,
    return mean, std

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, mean, std):
        # construct data with given mean and std
        if isinstance(std, torch.Tensor):
            std = std[None, :, None, None]
        if isinstance(mean, torch.Tensor):
            mean = mean[None, :, None, None]
        self.data = torch.randn(4000, 3, 24, 24) * std + mean
        
    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)

def intrinsic_matrix(camera):
    '''
    def shape_from_bpy(bpy_render=None):
        render = bpy_render or bpy.context.scene.render
        scale = render.resolution_percentage / 100.0
        shape = (
            int(render.resolution_y * scale),
            int(render.resolution_x * scale)
        )
        return shape
    ''' # if bpy_camera!
    h, w = camera.shape()  # btb.Camera

    focalLength = camera.data.lens
    sensorWidth = camera.data.sensor_width
    sensorHeight = camera.data.sensor_height
    # w / sensorWidth = pixels per mm; [1/mm]
    # focalLength is in units of mm!
    fx = focalLength * w / sensorWidth
    #fy = focalLength * h / sensorHeight
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    # more likely I need a numpy array instead 
    from mathutils import Matrix
    K = Matrix()
    K[0][0] = fx
    K[1][1] = fy
    K[0][2] = cx
    K[1][2] = cy
    K[3][3] = 0.0
    return K

def set_camera_intrinsics(scene, camera, shape, focallength_mm, chip_width_mm, principal_point):
    scene.render.resolution_x = shape[1]
    scene.render.resolution_y = shape[0]
    scene.render.resolution_percentage = 100
    # Shift is relative to largest dimension
    # https://blender.stackexchange.com/questions/58235/what-are-the-units-for-camera-shift
    # Also note that x and y behave differently.
    maxd = max(shape[1], shape[0])
    shift_x = -(principal_point[0] - shape[1]/2)/maxd
    shift_y = (principal_point[1] - shape[0]/2)/maxd
    camera.data.lens = focallength_mm
    camera.data.lens_unit = 'MILLIMETERS'
    camera.data.sensor_width = chip_width_mm
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.shift_x = shift_x
    camera.data.shift_y = shift_y

if __name__ == '__main__':
    mean, std = torch.tensor([2, 3, 4]), torch.tensor([5, 6, 7])
    td = TestDataset(mean, std)
    tdl = torch.utils.data.DataLoader(td, batch_size=1000,
        num_workers=4, shuffle=False)
    m, s = data_mean_and_std(tdl)
    print(m, s)

    m = td.data.mean([0, 2, 3])
    mu = m[None, :, None, None]
    s = torch.sqrt(torch.square(td.data - mu).mean([0, 2, 3]))
    print(m, s)

    '''
    tensor([2.0028, 3.0029, 3.9917]) tensor([4.9982, 5.9990, 6.9966])
    tensor([2.0028, 3.0029, 3.9917]) tensor([4.9982, 5.9990, 6.9966])
    '''