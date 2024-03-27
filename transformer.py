import torch
import torch.optim
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# import spectral as spy
from einops import rearrange
from thop import profile
torch.backends.cudnn.enabled = False
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from visualizer import get_local

dtype = torch.cuda.FloatTensor
MAEloss = torch.nn.L1Loss(reduction='mean').type(dtype)

def kaiming_init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.0)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activate="relu",
                 bn=None, pad_model=None, dilation=1, groups=1):
        super(Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activate = activate
        self.bn = bn
        self.pad_model = pad_model
        self.dilation = dilation
        self.groups = groups

        if self.bn == 'bn':
            self.batch = nn.BatchNorm2d(self.out_channels)
        elif self.bn == 'in':
            self.batch = nn.InstanceNorm2d(self.out_channels)
        else:
            self.batch = None

        if activate == "lrelu":
            self.act = nn.LeakyReLU(0.2, True)
        elif activate == "tanh":
            self.act = nn.Tanh()
        elif activate == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activate == 'relu':
            self.act = nn.ReLU(True)
        elif activate == 'prelu':
            self.act = nn.PReLU(self.out_channels, init=0.5)
        elif activate == 'logsigmoid':
            self.act = nn.LogSigmoid()
        elif activate == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = None

        if self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups)
        else:
            self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups)

        layers = filter(lambda x: x is not None, [self.conv, self.batch, self.act])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.pad_model is not None:
            x = self.padding(x)

        x = self.layers(x)

        return x


class Residual_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size=3, stride=1, padding=1, activate='relu', bn=False, scale=1, pad_model=None):
        super(Residual_Block, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = self.kernel_size // 2
        self.activate = activate
        self.bn = bn
        self.scale = scale
        self.pad_model = pad_model

        self.conv1 = Conv(self.in_channels, self.mid_channels, self.kernel_size, self.stride, self.padding,
                          activate='relu')
        self.conv2 = Conv(self.mid_channels, self.mid_channels, self.kernel_size, self.stride, self.padding,
                          activate='relu')
        self.conv3 = Conv(self.mid_channels, self.in_channels, self.kernel_size, self.stride, self.padding,
                          self.activate, self.bn, self.pad_model)

        layers = filter(lambda x: x is not None, [self.conv1, self.conv2, self.conv3])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        skip = x
        x = self.layers(x)
        x = x * self.scale
        x = torch.add(x, skip)

        return x

class Conv_up(nn.Module):
    def __init__(self, c_in, mid_c):
        super(Conv_up, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=mid_c, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        modules_tail = [
            nn.Upsample(scale_factor=4),
            nn.Conv2d(mid_c, c_in, 3, padding=(3 // 2), bias=True),
            nn.Conv2d(c_in, c_in, 3, padding=(3 // 2), bias=True)
        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):
        out = self.body(input)
        out = self.tail(out)

        return out

class Conv_down(nn.Module):
    def __init__(self, c_in, mid_c):
        super(Conv_down, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=mid_c, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        modules_tail = [
            nn.MaxPool2d(4),
            nn.Conv2d(mid_c, c_in, 3, padding=(3 // 2), bias=True),
            nn.Conv2d(c_in, c_in, 3, padding=(3 // 2), bias=True)
        ]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):
        out = self.body(input)
        out = self.tail(out)

        return out

class unet(nn.Module):
    def __init__(self, ms_channels):
        super(unet, self).__init__()

        self.ms_channels = ms_channels

        self.down1 = nn.Sequential(
            Conv(self.ms_channels + 1, 64),
            Residual_Block(64)
        )

        self.down2 = nn.Sequential(
            Conv(64, 128, kernel_size=4, stride=2, padding=1),
            Residual_Block(128)
        )

        self.layer = nn.Sequential(
            Conv(128, 256, kernel_size=4, stride=2, padding=1),
            Residual_Block(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.up2 = nn.Sequential(
            Conv(256, 128),
            Residual_Block(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.up1 = nn.Sequential(
            Conv(128, 64),
            Conv(64, self.ms_channels)
        )

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        layer = self.layer(down2)
        up2 = self.up2(torch.cat([layer, down2], 1))
        out = self.up1(torch.cat([up2, down1], 1))

        return out

class SpatialTransformer(nn.Module):
    def __init__(self, dim, heads):
        super(SpatialTransformer, self).__init__()

        self.dim = dim
        self.heads = heads
        head_dim = self.dim // self.heads

        self.rescale = head_dim ** -0.5

        self.to_q = nn.Sequential(
            Conv(self.dim, self.dim * self.heads, kernel_size=1, padding=0, activate=None),
            Conv(self.dim * self.heads, self.dim * self.heads, 3, 1, 3, activate=None, dilation=3)
        )
        self.to_k = nn.Sequential(
            Conv(self.dim, self.dim * self.heads, kernel_size=1, padding=0, activate=None),
            Conv(self.dim * self.heads, self.dim * self.heads, 3, 1, 2, activate=None, dilation=2)
        )
        self.to_v = nn.Sequential(
            Conv(self.dim, self.dim * self.heads, kernel_size=1, padding=0, activate=None),
            Conv(self.dim * self.heads, self.dim * self.heads, 3, 1, 1, activate=None, dilation=1)
        )

        self.proj = Conv(self.dim * self.heads, self.dim, 1, 1, 0, activate=None)

    # def visualize_head(self, att_map):
    #     ax = plt.gca()
    #     # Plot the heatmap
    #     im = ax.imshow(att_map)
    #     plt.axis('off')
    #     # Create colorbar
    #     # cbar = ax.figure.colorbar(im, ax=ax)
    #     plt.savefig('heatmap1.pdf', bbox_inches='tight',dpi=300,pad_inches=0.0)
    #     plt.show()

    @get_local('att')
    def forward(self, x):
        b, c, h, w = x.shape
        patch_size = 16
        nh = h // patch_size
        nw = w // patch_size

        q_in = self.to_q(x)
        k_in = self.to_k(x)
        v_in = self.to_v(x)

        q, k, v = map(lambda y: rearrange(y, 'b (head c) (nh h) (nw w) -> b (nh nw head) (h w) c', head=self.heads, h=patch_size, w=patch_size), (q_in, k_in, v_in))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        att1 = q @ k.transpose(-2, -1) * self.rescale
        att = att1.softmax(-1)

        # print(att[0, 1].cpu().numpy().max())
        # print(att[0, 1].cpu().numpy().min())
        # self.visualize_head(att[0, 0].cpu().numpy())
        # # Convert to NumPy arrays
        # attention_map_scipy = np.array(att[0,1,110:120,110:120].cpu())
        #
        # # Visualize
        # heat_map = sns.heatmap(attention_map_scipy, annot=True, cmap="YlGnBu")
        # picture1 = heat_map.get_figure()  # 得到生成的热力图
        # picture1.savefig("Heatmap.jpg", dpi=300, bbox_inches="tight")

        out = att @ v
        out = rearrange(out, 'b (nh nw head) (h w) c -> b (head c) (nh h) (nw w)', h=patch_size, nh=nh, nw=nw)
        out = self.proj(out)

        return out


class SpectralTransformer(nn.Module):
    def __init__(self, channels, heads):
        super(SpectralTransformer, self).__init__()

        self.channels = channels
        self.heads = heads

        # self.to_q = nn.Linear(self.channels, self.channels * self.heads, bias=False)
        # self.to_k = nn.Linear(self.channels, self.channels * self.heads, bias=False)
        # self.to_v = nn.Linear(self.channels, self.channels * self.heads, bias=False)
        self.to_q = nn.Sequential(
            Conv(self.channels, self.channels * self.heads, kernel_size=1, padding=0, activate=None),
            Conv(self.channels * self.heads, self.channels * self.heads, 3, 1, 3, activate=None, dilation=3, groups=self.channels * self.heads)
        )
        self.to_k = nn.Sequential(
            Conv(self.channels, self.channels * self.heads, kernel_size=1, padding=0, activate=None),
            Conv(self.channels * self.heads, self.channels * self.heads, 3, 1, 2, activate=None, dilation=2, groups=self.channels * self.heads)
        )
        self.to_v = nn.Sequential(
            Conv(self.channels, self.channels * self.heads, kernel_size=1, padding=0, activate=None),
            Conv(self.channels * self.heads, self.channels * self.heads, 3, 1, 1, activate=None, dilation=1, groups=self.channels * self.heads)
        )

        self.rescale = nn.Parameter(torch.ones(self.heads, 1, 1))

        self.proj = Conv(self.channels * self.heads, self.channels, kernel_size=1, padding=0, activate=None)

    # def visualize_head(self, att_map):
    #     ax = plt.gca()
    #     # Plot the heatmap
    #     im = ax.imshow(att_map)
    #     plt.axis('off')
    #     # Create colorbar
    #     # cbar = ax.figure.colorbar(im, ax=ax)
    #     plt.savefig('heatmap2.pdf', bbox_inches='tight',dpi=300,pad_inches=0.0)
    #     plt.show()

    def forward(self, x):
        b, c, h, w = x.shape

        # x_in = x.reshape(b, h * w, c)
        q_in = self.to_q(x)
        k_in = self.to_k(x)
        v_in = self.to_v(x)

        q, k, v = map(lambda y: rearrange(y, 'b (head c) h w -> b head c (h w)', head=self.heads), (q_in, k_in, v_in))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        att = q @ k.transpose(-2, -1) * self.rescale
        att = att.softmax(dim=-1)

        # self.visualize_head(att[0, 0].cpu().numpy())


        x_out = att @ v
        x_out = rearrange(x_out, 'b head c (h w) -> b (head c) h w', h=h)
        out = self.proj(x_out)

        return out

class FeedForward(nn.Module):
    def __init__(self, channels, dim=64):
        super(FeedForward, self).__init__()

        self.channels = channels
        self.dim = dim

        self.proj_in = Conv(self.channels, self.dim, 1, 1, 0, None)

        self.s1 = Conv(self.dim, self.dim, 3, 1, 1, 'gelu', dilation=1, groups=self.dim)
        self.s2 = Conv(self.dim, self.dim, 3, 1, 2, 'gelu', dilation=2, groups=self.dim)
        self.s3 = Conv(self.dim, self.dim, 3, 1, 3, 'gelu', dilation=3, groups=self.dim)

        self.proj_out = Conv(self.dim, self.channels, 1, 1, 0, None)

    def forward(self, x):
        x_in = self.proj_in(x)

        x1 = self.s1(x_in)
        x2 = self.s2(x_in)
        x3 = self.s3(x_in)

        out = self.proj_out(x1 + x2 + x3)

        return out


class HybridTransformer(nn.Module):
    def __init__(self, channels, heads):
        super(HybridTransformer, self).__init__()

        self.channels = channels
        self.heads = heads

        self.ln1 = nn.LayerNorm(self.channels)

        self.spatial = SpatialTransformer(self.channels, self.heads)
        self.spectral = SpectralTransformer(self.channels, self.heads)

        self.ln2 = nn.LayerNorm(self.channels)

        self.feedforward = FeedForward(self.channels)


    def forward(self, x):
        b, c, h, w = x.shape
        x_in = rearrange(x, 'b c h w -> b (h w) c')
        ln1 = self.ln1(x_in)
        x_in = rearrange(ln1, 'b (h w) c -> b c h w', h=h)
        middle = x + self.spatial(x_in) + self.spectral(x_in)

        middle_in = rearrange(middle, 'b c h w -> b (h w) c')
        ln2 = self.ln2(middle_in)
        middle_in = rearrange(ln2, 'b (h w) c -> b c h w', h=h)
        out = middle + self.feedforward(middle_in)

        return out


class mscsc(nn.Module):
    def __init__(self, ms_channels, pan_channel, iter_num=3, scale=3, dim=64, heads=2, num_blocks=3):
        super(mscsc, self).__init__()

        self.ms_channels = ms_channels
        self.pan_channel = pan_channel
        self.dim = dim
        self.iter_num = iter_num
        self.scale = scale
        self.heads = heads
        self.num_blocks = num_blocks

        self.hyperscale = nn.ModuleList()

        for i in range(self.iter_num):
            self.hyperscale.append(
                nn.Sequential(
                    Conv(self.ms_channels + self.pan_channel, self.dim, kernel_size=1, padding=0, activate=None),
                    Conv(self.dim, self.dim, activate=None),
                    Conv(self.dim,  2 * self.scale + 1, kernel_size=1, padding=0, activate=None),
                    nn.Softplus()
                )
            )

        self.x0 = nn.ModuleList()
        self.u0 = nn.ModuleList()
        self.filter = nn.ModuleList()
        self.defilter = nn.ModuleList()
        self.input_proj = nn.ModuleList()
        self.output_proj = nn.ModuleList()

        for s in range(self.scale):
            self.x0.append(
                nn.Sequential(
                    # Conv(self.ms_channels + self.pan_channel, self.dim, 3, 1, s + 1, dilation=s + 1),
                    # Conv(self.dim, self.dim, 3, 1, s + 1, dilation=s + 1, groups=self.dim),
                    # Conv(self.dim, self.ms_channels, 3, 1, s + 1, dilation=s + 1)
                    Conv(self.ms_channels + self.pan_channel, self.dim, 2 * s + 3, 1, s + 1, dilation=1),
                    Conv(self.dim, self.dim, 2 * s + 3, 1, s + 1, dilation=1, groups=self.dim),
                    Conv(self.dim, self.ms_channels, 2 * s + 3, 1, s + 1, dilation=1)
                )
            )

            self.u0.append(
                nn.Sequential(
                    HybridTransformer(self.ms_channels + self.pan_channel, 1),
                    Conv(self.ms_channels + self.pan_channel, self.ms_channels, 3, 1, 1, None)
                )
            )

            self.filter.append(
                nn.Sequential(
                    # Conv(self.ms_channels, self.dim, 3, 1, s + 1, dilation=s + 1),
                    # Conv(self.dim, self.dim, 3, 1, s + 1, dilation=s + 1, groups=self.dim),
                    # Conv(self.dim, self.ms_channels, 3, 1, s + 1, dilation=s + 1)
                    Conv(self.ms_channels, self.dim, 2 * s + 3, 1, s + 1, dilation=1),
                    Conv(self.dim, self.dim, 2 * s + 3, 1, s + 1, dilation=1, groups=self.dim),
                    Conv(self.dim, self.ms_channels, 2 * s + 3, 1, s + 1, dilation=1)
                )
            )

            self.defilter.append(
                nn.Sequential(
                    # nn.ConvTranspose2d(self.ms_channels, self.dim, 3, 1, s + 1, dilation=s + 1),
                    # nn.ConvTranspose2d(self.dim, self.dim, 3, 1, s + 1, groups=self.dim, dilation=s + 1),
                    # nn.ConvTranspose2d(self.dim, self.ms_channels, 3, 1, s + 1, dilation=s + 1)
                    nn.ConvTranspose2d(self.ms_channels, self.dim, 2 * s + 3, 1, s + 1, dilation=1),
                    nn.ConvTranspose2d(self.dim, self.dim, 2 * s + 3, 1, s + 1, groups=self.dim, dilation=1),
                    nn.ConvTranspose2d(self.dim, self.ms_channels, 2 * s + 3, 1, s + 1, dilation=1)
                )
            )

            self.input_proj.append(
                nn.Sequential(
                    Conv(self.ms_channels + 1, self.dim, activate=None)
                )
            )

            self.output_proj.append(
                nn.Sequential(
                    Conv(self.dim, self.ms_channels, activate=None)
                )
            )

        self.blocks = nn.ModuleList()

        for _ in range(self.num_blocks):
            self.blocks.append(HybridTransformer(self.dim, self.heads))

        self.A = nn.Sequential(
                Conv(self.ms_channels, self.dim),
                Conv(self.dim, self.dim),
                Conv(self.dim, self.pan_channel),
            )

        self.AT = nn.Sequential(
                Conv(self.pan_channel, self.dim),
                Conv(self.dim, self.dim),
                Conv(self.dim, self.ms_channels),
            )

        self.B = Conv_down(self.ms_channels, self.dim)
        self.BT = Conv_up(self.ms_channels, self.dim)

        kaiming_init_weights(self.u0, self.A, self.AT, self.B, self.BT, self.hyperscale, self.x0, self.filter, self.defilter)


    def conv_map(self, map):
        out = self.filter[0](map[0])
        out_plt = [out]
        for i in range(1, self.scale):
            output = self.filter[i](map[i])
            out_plt.append(output)
            out = out + output
        return out, out_plt


    def forward(self, lrms, pan, usms):
        xt = []
        ut = []

        for s in self.x0:
            xt.append(s(torch.cat([usms, pan], 1)))

        for u in self.u0:
            ut.append(u(torch.cat([usms, pan], 1)))

        for i in range(self.iter_num):
            hyperscale = self.hyperscale[i](torch.cat([usms, pan], 1))
            x_sum, _ = self.conv_map(xt)
            for j in range(self.scale):
                xt[j] = xt[j] - hyperscale[:, j, :, :].unsqueeze(1) * (self.defilter[j](self.AT(self.A(x_sum) - pan))
                        + self.defilter[j](self.BT(self.B(x_sum) - lrms))
                        + hyperscale[:, 2 * self.scale, :, :].unsqueeze(1) * (xt[j] - ut[j]))

                x_input = self.input_proj[j](torch.cat([xt[j], hyperscale[:, self.scale + j, :, :].unsqueeze(1)], 1))
                for b in self.blocks:
                    x_input = b(x_input)
                ut[j] = xt[j] + self.output_proj[j](x_input)

                for layer in self.filter[j]:
                    # print(layer.state_dict()['conv.weight'])
                    # if isinstance(layer, nn.Conv2d):
                    conv1_weight = layer.state_dict()['conv.weight'].cpu().data.numpy()
                    conv1_weight = (conv1_weight - conv1_weight.min()) / (
                                conv1_weight.max() - conv1_weight.min())
                # weight = self.filter[j].weight.data.numpy()

                # plt.figure(figsize=(100, 100))
                # plt.subplots_adjust(wspace=0, hspace=0)  # 去除子图之间的间隔
                # for k in range(conv1_weight.shape[0]):
                #     plt.subplot(16, 16, k + 1)
                #     plt.imshow(conv1_weight[k, :, :], cmap='gray')
                #     plt.axis('off')
                # ax = plt.gca()
                # Plot the heatmap
                # im = ax.imshow(weight)
                # plt.axis('off')
                # Create colorbar
                # cbar = ax.figure.colorbar(im, ax=ax)
                # plt.tight_layout()
                # # plt.savefig(f'weight_{j}.pdf', format='pdf', bbox_inches='tight')
                # plt.savefig(f'weight_{j}.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
                # plt.show()


        out, out_plt = self.conv_map(ut)
        return out, out_plt, ut

if __name__ == '__main__':
    net = 'trans'
    iter_num = 5
    scale = 3
    dim = 64
    heads = 2
    num_blocks = 3
    qb_channels = 4
    wv_channels = 8
    pan_channel = 1
    batch_size = 6
    ms_reduce = 64
    ms_full = 128
    lms_reduce = 256
    lms_full = 512

    model = mscsc(qb_channels, pan_channel=pan_channel, iter_num=iter_num, scale=scale, dim=dim, heads=heads, num_blocks=num_blocks)
    input = (torch.randn(batch_size, 4, ms_reduce, ms_reduce), torch.randn(batch_size, 1, 256, 256), torch.randn(batch_size, 4, lms_reduce, lms_reduce))
    flops, params = profile(model, (input[0], input[1],input[2],))
    print(f'{net} i5 QB_reduce FLOPS={flops / 1e9:.4f}G params={params / 1e6:.4f}M')
    model = mscsc(qb_channels, pan_channel=pan_channel, iter_num=iter_num, scale=scale, dim=dim, heads=heads, num_blocks=num_blocks)
    input = (torch.randn(batch_size, 4, ms_full, ms_full), torch.randn(batch_size, 1, 512, 512), torch.randn(batch_size, 4, lms_full, lms_full))
    flops, params = profile(model, (input[0], input[1],input[2],))
    print(f'{net} i5 QB_full FLOPS={flops / 1e9:.4f}G params={params / 1e6:.4f}M')

    model = mscsc(wv_channels, pan_channel=pan_channel, iter_num=iter_num, scale=scale, dim=dim, heads=heads, num_blocks=num_blocks)
    input = (torch.randn(batch_size, 8, ms_reduce, ms_reduce), torch.randn(batch_size, 1, 256, 256), torch.randn(batch_size, 8, lms_reduce, lms_reduce))
    flops, params = profile(model, (input[0], input[1],input[2],))
    print(f'{net} i5 wv3_reduce FLOPS={flops / 1e9:.4f}G params={params / 1e6:.4f}M')
    model = mscsc(wv_channels, pan_channel=pan_channel, iter_num=iter_num, scale=scale, dim=dim, heads=heads, num_blocks=num_blocks)
    input = (torch.randn(batch_size, 8, ms_full, ms_full), torch.randn(batch_size, 1, 512, 512), torch.randn(batch_size, 8, lms_full, lms_full))
    flops, params = profile(model, (input[0], input[1],input[2],))
    print(f'{net} i5 wv3_full FLOPS={flops / 1e9:.4f}G params={params / 1e6:.4f}M')

    iter_num = 3

    model = mscsc(qb_channels, pan_channel=pan_channel, iter_num=iter_num, scale=scale, dim=dim, heads=heads, num_blocks=num_blocks)
    input = (torch.randn(batch_size, 4, ms_reduce, ms_reduce), torch.randn(batch_size, 1, 256, 256), torch.randn(batch_size, 4, lms_reduce, lms_reduce))
    flops, params = profile(model, (input[0], input[1],input[2],))
    print(f'{net} i3 QB_reduce FLOPS={flops / 1e9:.4f}G params={params / 1e6:.4f}M')
    model = mscsc(qb_channels, pan_channel=pan_channel, iter_num=iter_num, scale=scale, dim=dim, heads=heads, num_blocks=num_blocks)
    input = (torch.randn(batch_size, 4, ms_full, ms_full), torch.randn(batch_size, 1, 512, 512), torch.randn(batch_size, 4, lms_full, lms_full))
    flops, params = profile(model, (input[0], input[1],input[2],))
    print(f'{net} i3 QB_full FLOPS={flops / 1e9:.4f}G params={params / 1e6:.4f}M')

    model = mscsc(wv_channels, pan_channel=pan_channel, iter_num=iter_num, scale=scale, dim=dim, heads=heads, num_blocks=num_blocks)
    input = (torch.randn(batch_size, 8, ms_reduce, ms_reduce), torch.randn(batch_size, 1, 256, 256), torch.randn(batch_size, 8, lms_reduce, lms_reduce))
    flops, params = profile(model, (input[0], input[1],input[2],))
    print(f'{net} i3 wv3_reduce FLOPS={flops / 1e9:.4f}G params={params / 1e6:.4f}M')
    model = mscsc(wv_channels, pan_channel=pan_channel, iter_num=iter_num, scale=scale, dim=dim, heads=heads, num_blocks=num_blocks)
    input = (torch.randn(batch_size, 8, ms_full, ms_full), torch.randn(batch_size, 1, 512, 512), torch.randn(batch_size, 8, lms_full, lms_full))
    flops, params = profile(model, (input[0], input[1],input[2],))
    print(f'{net} i3 wv3_full FLOPS={flops / 1e9:.4f}G params={params / 1e6:.4f}M')
