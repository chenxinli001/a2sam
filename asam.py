import numpy as np
import torch.nn as nn
import torch
from segment_anything import sam_model_registry
from sam_lora_image_encoder import LoRA_Sam
from utils import init_weights,init_weights_orthogonal_normal
import torch.nn.functional as F
from torch.distributions import Normal, Independent

#导入一些必须的类
class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers,padding=True,posterior=False,object=False,dataset='lidc'):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.posterior=posterior
        self.object=object
        if dataset=='lidc':
            if self.posterior and self.object:
                self.input_channels += 1
        else:
            if self.posterior and self.object:
                self.input_channels += 3
            elif self.posterior==False and self.object:
                self.input_channels += 2

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output



class AxisAlignedConvGaussian_box(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False,object=False,dataset='lidc'):
        super(AxisAlignedConvGaussian_box, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.output_channels=8
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.object=object
        self.dataset=dataset
        if self.posterior:
            self.name = 'Posterior'
            self.input_feature=1024
        else:
            self.name = 'Prior'
            self.input_feature=512
        self.box_input_channel=264
        self.fc1 = nn.Linear(self.input_feature, 1024) 
        self.fc2 = nn.Linear(1024, 512)             
        self.fc3 = nn.Linear(512, self.output_channels * 64)  
        self.box_conv = nn.Sequential(
            nn.Conv2d(self.box_input_channel, 128, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 8x8 -> 16x16
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 16x16 -> 16x16
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 16x16 -> 32x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32x32 -> 32x32
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 32x32 -> 64x64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 64x64 -> 64x64
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 64x64 -> 128x128
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # 128x128 -> 128x128
        )


        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers,posterior=self.posterior, object=self.object,dataset=self.dataset)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input,boxemb_shift,boxemb_ori=None):
        if boxemb_ori is not None:
            boxemb_input = torch.cat((boxemb_shift, boxemb_ori), dim=1)
        else:
            boxemb_input = boxemb_shift
        
        boxemb_input = boxemb_input.view(boxemb_input.size(0), -1)
        boxemb_input = F.relu(self.fc1(boxemb_input))
        boxemb_input = F.relu(self.fc2(boxemb_input))
        boxemb_input = self.fc3(boxemb_input)

        boxemb_input = boxemb_input.view(boxemb_input.size(0), self.output_channels, 8, 8)
        input=torch.cat((input, boxemb_input), dim=1)
        input = input.to(torch.float32)
        input=self.box_conv(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image，
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        mu_log_sigma = self.conv_layer(encoding)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        
        return dist


class Fcomb_box(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb_box, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [1,2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(512, 256, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(256, 256, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(256, 256, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        # 确保 order_index 在与 a 相同的设备上
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
        return torch.index_select(a, dim, order_index)


    def forward(self, feature_map, z):
        if self.use_tile:
            # print(feature_map.shape)#torch.Size([1, 256, 8, 8])
            # print(z.shape)#torch.Size([1, 6])

            z = torch.unsqueeze(z,2)
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[1]])
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[2]])
            feature_map = torch.cat((feature_map, z), dim=1)
            output = self.layers(feature_map)
            output = self.last_layer(output)
            return output     




class AxisAlignedConvGaussian_object(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False,object=False,dataset='lidc'):
        super(AxisAlignedConvGaussian_object, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.object=object
        self.dataset=dataset
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers,posterior=self.posterior, object=self.object,dataset=self.dataset)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input,segm=None):
        if segm is not None:
            input = torch.cat((input, segm), dim=1)

        input = input.to(torch.float32)
        encoding = self.encoder(input)
        self.show_enc = encoding

        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)


        mu_log_sigma = self.conv_layer(encoding)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        
        return dist


class Fcomb_object(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb_object, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [1,2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(512, 256, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(256, 256, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(256, 256, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:

            z = torch.unsqueeze(z,2)#
            z = torch.unsqueeze(z,2)#
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[1]])
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[2]])
            feature_map = torch.cat((feature_map, z), dim=1)
            output = self.layers(feature_map)
            output = self.last_layer(output)

            return output     


class ASAM(nn.Module):

    def __init__(self,input_channels=1, num_classes=6, img_size=128,num_filters=[32,64,128,192],latent_dim=256, no_convs_fcomb=4, beta=10.0,dataset='lidc'):
        super(ASAM, self).__init__()
        self.ckpt="sam_vit_b_01ec64.pth"
        self.img_size=img_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = beta
        self.z_prior_sample = 0
        self.dataset=dataset
        self.sam,self.img_embedding_size = sam_model_registry["vit_b"](image_size=self.img_size,
                                            num_classes=self.num_classes,
                                            checkpoint=self.ckpt, pixel_mean=[0, 0, 0],
                                            pixel_std=[1, 1, 1])
        self.lora_sam=LoRA_Sam(self.sam,4)
        self.prior_object = AxisAlignedConvGaussian_object(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers,posterior=False,object=True,dataset=self.dataset)
        self.prior_box = AxisAlignedConvGaussian_box(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers,posterior=False,object=False,dataset=self.dataset)
        self.posterior_object = AxisAlignedConvGaussian_object(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers,posterior=True,object=True,dataset=self.dataset)
        self.posterior_box = AxisAlignedConvGaussian_box(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers,posterior=True,object=False,dataset=self.dataset)
        self.fcomb_object = Fcomb_object(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True)
        self.fcomb_box= Fcomb_box(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True)


    
    def forward(self,batch_input,batch_input_ori,batch_boxori,batch_boxshift,batch_mask,device,input_size=128,train=True):
        img_size = input_size
        self.lora_sam.sam.to(device)
        self.prior_object.to(device)
        self.prior_box.to(device)
        self.posterior_object.to(device)
        self.posterior_box.to(device)
        self.fcomb_object.to(device)
        self.fcomb_box.to(device)
        input_images = self.lora_sam.sam.preprocess(batch_input)
        image_embeddings = self.lora_sam.sam.image_encoder(input_images)      
        sparse_embeddings_shift, dense_embeddings_shift = self.lora_sam.sam.prompt_encoder(
            points=None, boxes=batch_boxshift, masks=None
        )
        sparse_embeddings_ori, dense_embeddings_ori = self.lora_sam.sam.prompt_encoder(
            points=None, boxes=batch_boxori, masks=None
        )
        self.prior_box_latent_space = self.prior_box.forward(image_embeddings,sparse_embeddings_shift)
        self.prior_object_latent_space = self.prior_object.forward(batch_input_ori)
        if train:
            self.posterior_box_latent_space = self.posterior_box.forward(image_embeddings, sparse_embeddings_ori,sparse_embeddings_shift)
            self.posterior_object_latent_space = self.posterior_object.forward(batch_input_ori, batch_mask.unsqueeze(1))
            self.z_posterior_box = self.posterior_box_latent_space.rsample()
            self.z_posterior_object = self.posterior_object_latent_space.rsample()
            self.z_prior_box=self.prior_box_latent_space.rsample()
            self.z_prior_object=self.prior_object_latent_space.rsample()
            dense_embeddings_disturb=self.fcomb_box.forward(dense_embeddings_shift,self.z_posterior_box)
            image_embeddings_disturb=self.fcomb_object.forward(image_embeddings,self.z_posterior_object)
            

            
        else:
            self.z_prior_box=self.prior_box_latent_space.sample()
            self.z_prior_object=self.prior_object_latent_space.sample()
            dense_embeddings_disturb=self.fcomb_box.forward(dense_embeddings_shift,self.z_prior_box)
            image_embeddings_disturb=self.fcomb_object.forward(image_embeddings,self.z_prior_object)



        low_res_masks, iou_predictions = self.lora_sam.sam.mask_decoder(
            image_embeddings=image_embeddings_disturb,
            image_pe=self.lora_sam.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings_shift,
            dense_prompt_embeddings=dense_embeddings_disturb,
            multimask_output=True
        )
        masks = self.lora_sam.sam.postprocess_masks(
            low_res_masks,
            input_size=(img_size,img_size ),
            original_size=(128, 128)
        )
      
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }
        return outputs


