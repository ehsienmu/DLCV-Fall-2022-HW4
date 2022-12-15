_base_ = '../default.py'

expname = 'dvgo_hotdog'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/home/hsien/dlcv/hw4-ehsienmu/hw4_data/hotdog',
    # dataset_type='dlcv',
    dataset_type='blender',
    white_bkgd=True,
)

dlcv_hw4_out = './dlcv_out'

# _base_ = '../default.py'

# expname = 'dvgo_hotdog'
# basedir = './logs/nerf_synthetic'

# data = dict(
#     datadir='./data/nerf_synthetic/hotdog',
#     dataset_type='blender',
#     white_bkgd=True,
# )

