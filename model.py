import torch
import torch.nn as nn
import torch.nn.functional as F
from pvt_v2 import pvt_v2_b2
from torchvision.ops import DeformConv2d
from torchvision.utils import save_image


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bn=True, relu=True):
        super().__init__()
        if bn:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=True)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, 3, padding=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, 3, padding=1),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, 3, padding=1),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.conv_cat = BasicConv2d(3 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))
        x = F.relu(x_cat + self.conv_res(x), inplace=True)
        return x


class BCA(nn.Module):
    def __init__(self, in_channels, mid_channels, scale=1):
        super(BCA, self).__init__()
        self.mid_channels = mid_channels
        self.scale = scale
        self.fself = BasicConv2d(in_channels, mid_channels, 1, bn=False, relu=False)
        self.fx = nn.Linear(in_channels, mid_channels)
        self.fy = nn.Linear(in_channels, mid_channels)
        self.spatial_conv = BasicConv2d(2, 1, 7, padding=3, bn=False, relu=False)

    def forward(self, x, y):
        batch_size = x.size(0)
        fself = self.fself(x).view(batch_size, self.mid_channels, -1)
        fx = self.fx(x.mean(dim=(2, 3))).view(batch_size, self.mid_channels, -1)        # (B, C, 1)  query
        fy = self.fy(y.mean(dim=(2, 3))).view(batch_size, self.mid_channels, -1)        # (B, C, 1)  key
        fy = fy.permute(0, 2, 1).contiguous()
        sim_map = torch.bmm(fx, fy)
        sim_map_div_C = torch.max(sim_map, -1, keepdim=True)[0].expand_as(sim_map) - sim_map
        sim_map_div_C = F.softmax(sim_map_div_C, dim=-1)
        fout = torch.bmm(sim_map_div_C, fself)
        fout = fout.view(batch_size, self.mid_channels, *x.size()[2:])
        if self.scale < 1:
            y = F.interpolate(y, scale_factor=self.scale, mode='bilinear', align_corners=False)
        spatial_map = torch.cat([torch.mean(x, 1).unsqueeze(1), torch.mean(y, 1).unsqueeze(1)], 1)
        spatial_map = self.spatial_conv(spatial_map)
        spatial_map = torch.sigmoid(spatial_map)
        return fout + fout * spatial_map


class BGFA(nn.Module):
    def __init__(self, channel, scale=1):
        super().__init__()
        self.scale = scale
        self.conv_offset = BasicConv2d(2 * channel, 18, 3, padding=1, bn=False, relu=False)
        self.conv_mask = BasicConv2d(2 * channel, 9, 3, padding=1, bn=False, relu=False)
        self.deform_conv = DeformConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv_weight_h = BasicConv2d(channel, 1, 1, relu=False)
        self.conv_weight_l = BasicConv2d(channel, 1, 1, relu=False)

    def forward(self, xl, xh, e):
        xl = F.interpolate(xl, size=xh.size()[2:], mode='bilinear', align_corners=False)
        if self.scale < 1:
            e = F.interpolate(e, scale_factor=self.scale, mode='bilinear', align_corners=False)
        query = torch.cat((xl + e, xh + e), 1)
        offset = self.conv_offset(query)
        mask = self.conv_mask(query)
        weight_h = self.conv_weight_h(xh)
        weight_l = self.conv_weight_l(xl)
        weight_h, weight_l = F.softmax(torch.cat((weight_h, weight_l), 1), 1).chunk(2, 1)
        xh = weight_l * self.deform_conv(xl, offset, mask=mask) + weight_h * xh
        return xh
class FaPN(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.bgfa1 = BGFA(channel, 1)
        self.bgfa2 = BGFA(channel, 0.5)
        self.bgfa3 = BGFA(channel, 0.25)

    def forward(self, x1_rfb, x2_rfb, x3_rfb, x4_rfb, e):
        x3_rfb = self.bgfa3(x4_rfb, x3_rfb, e)
        x2_rfb = self.bgfa2(x3_rfb, x2_rfb, e)
        x1_rfb = self.bgfa1(x2_rfb, x1_rfb, e)
        return x1_rfb, x2_rfb, x3_rfb, x4_rfb

class MyPolypPVT(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
        self.backbone = pvt_v2_b2(pretrained=True)
        self.rfb1 = RFB(64, channel)
        self.rfb2 = RFB(128, channel)
        self.rfb3 = RFB(320, channel)
        self.rfb4 = RFB(512, channel)

        self.fapn = FaPN(channel)

        self.conv_edge1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.conv_edge2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.conv_edge3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.conv_edge4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
        )

        self.conv_edge1_out = BasicConv2d(channel, 1, 1, bn=False, relu=False)
        self.conv_edge2_out = BasicConv2d(channel, 1, 1, bn=False, relu=False)
        self.conv_edge3_out = BasicConv2d(channel, 1, 1, bn=False, relu=False)
        self.conv_edge4_out = BasicConv2d(channel, 1, 1, bn=False, relu=False)

        self.shrink = BasicConv2d(4 * channel, channel, 1, bn=False, relu=False)

        self.bca1 = BCA(channel, channel, 1)
        self.bca2 = BCA(channel, channel, 0.5)
        self.bca3 = BCA(channel, channel, 0.25)
        self.bca4 = BCA(channel, channel, 0.125)

        self.seg_head1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, 1, 1, bn=False, relu=False),
        )
        self.seg_head2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, 1, 1, bn=False, relu=False),
        )
        self.seg_head3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, 1, 1, bn=False, relu=False),
        )
        self.seg_head4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, 1, 1, bn=False, relu=False),
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        x1_rfb = self.rfb1(x1)  # (N, 32, 88, 88)
        x2_rfb = self.rfb2(x2)  # (N, 32, 44, 44)
        x3_rfb = self.rfb3(x3)  # (N, 32, 22, 22)
        x4_rfb = self.rfb4(x4)  # (N, 32, 11, 11)
        e1 = self.conv_edge1(x1_rfb)
        e2 = self.conv_edge2(x2_rfb)
        e2 = F.interpolate(e2, size=e1.size()[2:], mode='bilinear', align_corners=False)
        e3 = self.conv_edge3(x3_rfb)
        e3 = F.interpolate(e3, size=e1.size()[2:], mode='bilinear', align_corners=False)
        e4 = self.conv_edge4(x4_rfb)
        e4 = F.interpolate(e4, size=e1.size()[2:], mode='bilinear', align_corners=False)

        e1_out = self.conv_edge1_out(e1)
        e2_out = self.conv_edge2_out(e2)
        e3_out = self.conv_edge3_out(e3)
        e4_out = self.conv_edge4_out(e4)
        e_out = e1_out + e2_out + e3_out + e4_out

        e = self.shrink(torch.cat((e1, e2, e3, e4), 1))

        x1_rfb, x2_rfb, x3_rfb, x4_rfb = self.fapn(x1_rfb, x2_rfb, x3_rfb, x4_rfb, e)
        x1_rfb = self.bca1(x1_rfb, e)
        x2_rfb = self.bca2(x2_rfb, e)
        x3_rfb = self.bca3(x3_rfb, e)
        x4_rfb = self.bca4(x4_rfb, e)
        out1 = self.seg_head1(x1_rfb)
        out2 = self.seg_head2(x2_rfb)
        out3 = self.seg_head3(x3_rfb)
        out4 = self.seg_head4(x4_rfb)
        return out1, out2, out3, out4, e_out
from thop import profile
torch.cuda.set_device(1)
if __name__ == '__main__':
    model = MyPolypPVT().cuda()
    x = torch.randn(1, 3, 352, 352).cuda()
    flops, params = profile(model, inputs=(x,))  # ���ص���һ��Ԫ��
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))

