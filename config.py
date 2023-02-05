################Model Tmplates##########################
#model = monai.networks.nets.resnet101(spatial_dims=3, n_input_channels=1, n_classes=1)
#model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, n_classes=1)
#model = monai.networks.nets.resnet34(spatial_dims=3, n_input_channels=1, n_classes=1)
#model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, n_classes=1)
#monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
#monai.networks.nets.DenseNet169(spatial_dims=3, in_channels=1, out_channels=1)
#monai.networks.nets.DenseNet209(spatial_dims=3, in_channels=1, out_channels=1)
#monai.networks.nets.DenseNet264(spatial_dims=3, in_channels=1, out_channels=1)
#monai.networks.nets.AHNet(spatial_dims=3, in_channels=1, out_channels=1)# Does not work
#monai.networks.nets.EfficientNetBN("efficientnet-b0",spatial_dims=3, in_channels=1, num_classes=1)
#monai.networks.nets.HighResNet(spatial_dims=3, in_channels=1, out_channels=1)# Does not work

################Import Block Start##########################
import monai
################Import Block END##########################

#-----------------------Config file DenseNet264 exp3 Start----------------------#
#####################Variables Block Start#######################
inp_dir="../Input/prep/exp1/"
out_dir="../Output/Script2/Results/"
label_file = "../Input/prep/exp1/train.csv"
NUM_IMAGES_3D = 64
TRAINING_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
IMAGE_SIZE = 256
N_EPOCHS = 100
do_valid = True
n_workers = 16
#####################Variables Block END#######################

##########################Functions Block Start###############################
#MODEL = monai.networks.nets.AHNet(spatial_dims=3, in_channels=1, out_channels=1)# DL model from MONAI
MODEL = monai.networks.nets.DenseNet264(spatial_dims=3, in_channels=1, out_channels=1)
##########################Functions Block END###############################


# #-----------------------Config file  End----------------------#
