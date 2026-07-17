import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange
from torchsummary import summary

class MDTA(nn.Module):
    def __init__(self, out_c, model_c):
        super(MDTA, self).__init__()
        self.norm = nn.GroupNorm(model_c, out_c)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1, 1, 0),
            nn.Conv2d(out_c, out_c, 3, 1, 1)
        )
        self.conv4 = nn.Conv2d(out_c, out_c, 1, 1, 0)
        
    def forward(self, x):
        x_o = x
        x = self.norm(x)
        C, W, H = x.size()[1], x.size()[2], x.size()[3]
        q = self.conv1(x)
        q = rearrange(q, 'b c w h -> b (w h) c')

        k = self.conv2(x)
        k = rearrange(k, 'b c w h -> b c (w h)')

        v = self.conv3(x)
        v = rearrange(v, 'b c w h -> b (w h) c')

        A = torch.matmul(k, q)
        A = rearrange(A, 'b c1 c2 -> b (c1 c2)', c1=C, c2=C)
        A = torch.softmax(A, dim=1)
        A = rearrange(A, 'b (c1 c2) -> b c1 c2', c1=C, c2=C)

        v = torch.matmul(v, A)
        v = rearrange(v, 'b (h w) c -> b c w h', c = C, h=H, w=W)

        return self.conv4(v) + x_o

class GDFN(nn.Module):
    def __init__(self, out_c, model_c):
        super(GDFN, self).__init__()
        self.norm = nn.GroupNorm(model_c, out_c)
        self.Dconv1 = nn.Sequential(
            nn.Conv2d(out_c, out_c*4, 1, 1, 0),
            nn.Conv2d(out_c*4, out_c*4, 3, 1, 1)
        )
        self.Dconv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c * 4, 1, 1, 0),
            nn.Conv2d(out_c * 4, out_c * 4, 3, 1, 1)
        )
        self.conv = nn.Conv2d(out_c * 4, out_c, 1, 1, 0)

    def forward(self, x):
        x_o = x
        x = self.norm(x)
        x = F.gelu(self.Dconv1(x)) * self.Dconv2(x)
        x = x_o + self.conv(x)
        return x


class Restormer(nn.Module):
    def __init__(self, in_c, out_c, model_c):
        super(Restormer, self).__init__()
        self.mlp = nn.Conv2d(in_c, out_c, 1, 1, 0)
        self.mdta = MDTA(out_c, model_c)
        self.gdfn = GDFN(out_c, model_c)
    def forward(self, feature):
        feature = self.mlp(feature)
        feature = self.mdta(feature)
        return self.gdfn(feature)
    

class TembRes(nn.Module):
    def __init__(self, in_channels, emb_channels, out_channels, model_c):
        super().__init__()
        self.norm1 = nn.GroupNorm(model_c, out_channels)
        self.norm2 = nn.GroupNorm(model_c, out_channels)
        self.fea_layers = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_channels, out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, emb):
        h = self.fea_layers(x)
        h = self.norm1(h)
        emb = self.emb_layers(emb)
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]
        h = h + emb
        out = self.out_layers(h)
        out = self.norm2(out)
        return out
    

class RestormerBackbone(nn.Module):
    def __init__(self, in_c, model_c, mult, emb_c):
        super(RestormerBackbone, self).__init__()
        self.channel = model_c
        self.mult = mult

        self.input = nn.Conv2d(in_c, self.channel, 1, 1, 0)

        self.restormers = nn.ModuleList([
            nn.Sequential(
                Restormer(self.mult[i]*self.channel, self.mult[i+1]*self.channel, model_c),
                nn.MaxPool2d(2, 2)
                )
            for i in range(len(self.mult) - 1)])
        
        self.TimeEmbs = nn.ModuleList([
            TembRes(self.mult[i+1]*self.channel, emb_c, self.mult[i+1]*self.channel, model_c)
            for i in range(len(self.mult) - 1)
        ])

    def forward(self, x, temb):
        middle = self.input(x)
        out = []
        for i in range(len(self.mult) - 1):
            module = self.restormers[i]
            t_embs = self.TimeEmbs[i]
            middle = module(middle)
            middle = t_embs(middle, temb)
            out.append(middle)
        return out
    

class TaskHead(nn.Module):
    def __init__(self, out_c, model_c, mult):
        super(TaskHead, self).__init__()

        self.num_mult = len(mult) - 2

        in_cs = [mult[-1]*model_c]
        for i in range(len(mult) - 2):
            in_cs.append(2*mult[-i-2]*model_c)
        out_cs = [mult[-i-2]*model_c for i in range(len(mult)-1)]

        self.out = nn.Sequential(
            Restormer(model_c, model_c, model_c),
            nn.Conv2d(model_c, out_c, 1, 1, 0),
        )

        self.restormers = nn.ModuleList([
            nn.Sequential(
                Restormer(in_cs[i], out_cs[i], model_c),
                nn.Upsample(scale_factor=2, mode='nearest')
                )
            for i in range(len(mult) - 1)])
        
    def forward(self, out):
        out_ = self.restormers[0](out[-1])
        for i in range(self.num_mult):
            out_ = torch.cat([out_, out[-i-2]], 1)
            out_ = self.restormers[i+1](out_)
        return self.out(out_)
    

class MultiTask(nn.Module):
    def __init__(self, in_c, out_c, model_c, mult, feat_c, emb_c):
        super(MultiTask, self).__init__()

        self.Backbone = RestormerBackbone(in_c, model_c, mult, emb_c)
        self.x0Head = TaskHead(out_c, model_c, mult)
        self.featHead = TaskHead(feat_c, model_c, mult)

    def forward(self, x, time_emb, context):
        x = torch.cat([x, context], 1)
        out = self.Backbone(x, time_emb)
        x0_recon = self.x0Head(out)
        feature = self.featHead(out)
        return x0_recon, feature
    
    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))


class ResUnet(nn.Module):
    def __init__(self, in_c, out_c, model_c, mult, emb_c):
        super(ResUnet, self).__init__()

        self.Backbone = RestormerBackbone(in_c, model_c, mult, emb_c)
        self.x0Head = TaskHead(out_c, model_c, mult)

    def forward(self, x, time_emb, context):
        x = torch.cat([x, context], 1)
        out = self.Backbone(x, time_emb)
        x0_recon = self.x0Head(out)
        return x0_recon
    
    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))


if __name__ == '__main__':
    in_c = 5
    out_c = 5
    model_c = 8
    mult = [1, 2, 4, 8]
    feat_c = 5
    emb_c = 64
    device = torch.device('cuda')
    Model = ResUnet(in_c, out_c, model_c, mult, emb_c).to(device)
    '''
    a = torch.ones([4, 5, 288, 320]).to(device)
    t = torch.ones([4, emb_c]).to(device)
    out1, out2 = Model(a, t)
    print(out1.shape)
    '''
    Model.get_parameter_number()
