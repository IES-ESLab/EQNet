from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

default_cfgs = {}


def log_transform(x):
    x = torch.sign(x) * torch.log(1.0 + torch.abs(x))
    return x


def moving_normalize(data, filter=1024, stride=128):
    nb, nch, nx, nt = data.shape

    # if nt % stride == 0:
    #     pad = max(filter - stride, 0)
    # else:
    #     pad = max(filter - (nt % stride), 0)
    # pad1 = pad // 2
    # pad2 = pad - pad1
    padding = filter // 2

    with torch.no_grad():
        # data_ = F.pad(data, (pad1, pad2, 0, 0), mode="reflect")
        data_ = F.pad(data, (padding, padding, 0, 0), mode="reflect")
        mean = F.avg_pool2d(data_, kernel_size=(1, filter), stride=(1, stride))
        mean = F.interpolate(mean, scale_factor=(1, stride), mode="bilinear", align_corners=False)[:, :, :nx, :nt]
        data -= mean

        # data_ = F.pad(data, (pad1, pad2, 0, 0), mode="reflect")
        data_ = F.pad(data, (padding, padding, 0, 0), mode="reflect")
        # std = (F.lp_pool2d(data_, norm_type=2, kernel_size=(filter, 1), stride=(stride, 1)) / ((filter) ** 0.5))
        std = F.avg_pool2d(torch.abs(data_), kernel_size=(1, filter), stride=(1, stride))
        std = torch.mean(std, dim=(1,), keepdim=True)  ## keep relative amplitude between channels
        std = F.interpolate(std, scale_factor=(1, stride), mode="bilinear", align_corners=False)[:, :, :nx, :nt]
        std[std == 0.0] = 1.0
        data = data / std

        # data = log_transform(data)

    return data


class MergeFrequency(nn.Module):
    """
    Merge frequency dimension to 1 using a linear layer.
    """

    def __init__(self, dim_in):
        super().__init__()
        # self.linear = nn.Sequential(nn.Linear(dim_in, 1), nn.ReLU())
        self.linear = nn.Linear(dim_in, 1)

    def forward(self, x):
        # x: nb, nc, nf, nt
        x = x.permute(0, 1, 3, 2)  # nb, nc, nt, nf
        x = self.linear(x).squeeze(-1)  # nb, nc, nt
        return x


class MergeBranch(nn.Module):
    """
    Merge two branches of the same dimension.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        # self.conv = nn.Sequential(nn.Conv2d(dim_in, dim_out, 1), nn.ReLU())
        self.conv = nn.Conv2d(dim_in, dim_out, 1)

    def forward(self, x1, x2):
        return self.conv(torch.cat((x1, x2), dim=1))


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=128 + 1,
        hop_length=4,
        window_fn=torch.hann_window,
        magnitude=True,
        normalize_freq=False,
        discard_zero_freq=True,
        **kwargs,
    ):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.magnitude = magnitude
        self.discard_zero_freq = discard_zero_freq
        self.normalize_freq = normalize_freq
        self.register_buffer("window", window_fn(n_fft))
        self.window_fn = window_fn

    def forward(self, x):
        # window = self.window_fn(self.n_fft).to(x.device)
        """
        x: bt, ch, nt
        """
        nb, nc, nt = x.shape
        x = x.view(-1, nt)  # nb*nc, nt
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            window=self.window,
            hop_length=self.hop_length,
            center=True,
            return_complex=True,
        )
        stft = torch.view_as_real(stft)
        # stft = stft[..., : x.shape[-1] // self.hop_length, :]  # nb*nc*nx, nf, nt, 2
        # stft = stft[..., :-1, :]
        if self.discard_zero_freq:
            stft = stft.narrow(dim=-3, start=1, length=stft.shape[-3] - 1)
        nf, nt, _ = stft.shape[-3:]
        if self.magnitude:
            stft = torch.norm(stft, dim=-1, keepdim=False).view(nb, nc, nf, nt)  # nb, nc, nf, nt
        else:
            stft = stft.view(nb, nc, nf, nt, 2)  # nb, nc, nf, nt, 2
            stft = stft.permute(0, 1, 4, 2, 3).view(nb, nc * 2, nf, nt)  # nb, nc*2, nf, nt

        if self.normalize_freq:
            vmax = torch.max(torch.abs(stft), dim=-2, keepdim=True)[0]
            vmax[vmax == 0.0] = 1.0
            stft = stft / vmax

        return stft


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        init_features=16,
        init_stride=(1, 1),
        kernel_size=(1, 7),
        stride=(1, 4),
        padding=(0, 3),
        moving_norm=(1024, 128),
        upsample="conv_transpose",
        add_stft=False,
        log_scale=False,
        add_polarity=False,
        add_event=False,
        add_prompt=False,
    ):
        super(UNet, self).__init__()

        features = init_features

        self.moving_norm = moving_norm
        self.log_scale = log_scale
        self.add_stft = add_stft
        self.add_polarity = add_polarity
        self.add_event = add_event
        self.add_prompt = add_prompt

        self.input_conv = self.encoder_block(
            in_channels, features, kernel_size=kernel_size, stride=init_stride, padding=padding, name="enc1"
        )
        self.encoder12 = self.encoder_block(
            features, features * 2, kernel_size=kernel_size, stride=stride, padding=padding, name="enc2"
        )
        self.encoder23 = self.encoder_block(
            features * 2, features * 4, kernel_size=kernel_size, stride=stride, padding=padding, name="enc3"
        )
        self.encoder34 = self.encoder_block(
            features * 4, features * 8, kernel_size=kernel_size, stride=stride, padding=padding, name="enc4"
        )
        self.encoder45 = self.encoder_block(
            features * 8, features * 16, kernel_size=kernel_size, stride=stride, padding=padding, name="enc5"
        )

        if upsample == "interpolate":
            self.upconv54 = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "bottle_conv",
                            nn.Conv2d(
                                in_channels=features * 16,
                                out_channels=features * 8,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=False,
                            ),
                        ),
                        ("bottle_norm", nn.BatchNorm2d(num_features=features * 8)),
                        ("bottle_relu", nn.ReLU(inplace=True)),
                        ("upsample", nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)),
                    ]
                )
            )
        elif upsample == "conv_transpose":
            self.upconv54 = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "bottle_conv",
                            nn.ConvTranspose2d(
                                in_channels=features * 16,
                                out_channels=features * 8,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                output_padding=padding,
                                bias=False,
                            ),
                        ),
                        ("bottle_norm", nn.BatchNorm2d(num_features=features * 8)),
                        ("bottle_relu", nn.ReLU(inplace=True)),
                    ]
                )
            )

        self.decoder43 = self.decoder_block(
            features * 16,
            features * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            upsample=upsample,
            name="dec4",
        )
        self.decoder32 = self.decoder_block(
            features * 8,
            features * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            upsample=upsample,
            name="dec3",
        )
        self.decoder21 = self.decoder_block(
            features * 4,
            features * 1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            upsample=upsample,
            name="dec2",
        )
        self.output_conv = self.encoder_block(
            features * 2, features * 2, kernel_size=kernel_size, stride=(1, 1), padding=padding, name="output"
        )

        if self.add_polarity:
            self.encoder_polarity = self.encoder_block(
                1, features, kernel_size=kernel_size, stride=(1, 1), padding=padding, name="enc1_polarity"
            )
            self.output_polarity = self.encoder_block(
                features * 2,
                features * 2,
                kernel_size=kernel_size,
                stride=(1, 1),
                padding=padding,
                name="output_polarity",
            )

        if self.add_event:
            self.output_event = self.encoder_block(
                features * 4, features * 2, kernel_size=kernel_size, stride=(1, 1), padding=padding, name="output_event"
            )

        if self.add_stft:
            self.n_fft = 64 + 1
            self.stft = STFT(
                n_fft=self.n_fft,
                hop_length=stride[-1],
                window_fn=torch.hann_window,
                magnitude=True,
                discard_zero_freq=True,
            )
            kernel_size_tf = (3, kernel_size[-1])  # 3 for frequency
            padding_tf = (1, padding[-1])
            self.encoder12_tf = self.encoder_block(
                in_channels,
                features,
                kernel_size=kernel_size_tf,
                stride=(1, 1),
                padding=padding_tf,
                name="enc2_tf",
            )
            self.encoder23_tf = self.encoder_block(
                features,
                features * 2,
                kernel_size=kernel_size_tf,
                stride=stride,
                padding=padding_tf,
                name="enc3_tf",
            )
            self.encoder34_tf = self.encoder_block(
                features * 2,
                features * 4,
                kernel_size=kernel_size_tf,
                stride=stride,
                padding=padding_tf,
                name="enc4_tf",
            )
            self.encoder45_tf = self.encoder_block(
                features * 4,
                features * 8,
                kernel_size=kernel_size_tf,
                stride=stride,
                padding=padding_tf,
                name="enc5_tf",
            )
            self.merge_freq2 = MergeFrequency(self.n_fft // 2)
            self.merge_freq3 = MergeFrequency(self.n_fft // 2)
            self.merge_freq4 = MergeFrequency(self.n_fft // 2)
            self.merge_freq5 = MergeFrequency(self.n_fft // 2)
            self.merge_branch2 = MergeBranch(features * 3, features * 2)
            self.merge_branch3 = MergeBranch(features * 6, features * 4)
            self.merge_branch4 = MergeBranch(features * 12, features * 8)
            self.merge_branch5 = MergeBranch(features * 24, features * 16)

        if (init_stride[0] > 1) or (init_stride[1] > 1):
            self.output_upsample = nn.Upsample(scale_factor=init_stride, mode="bilinear", align_corners=False)
        else:
            self.output_upsample = None

    def forward(self, x):

        x = moving_normalize(x, filter=self.moving_norm[0], stride=self.moving_norm[1])
        if self.log_scale:
            x = log_transform(x)

        # polarity
        if self.add_polarity:
            z = x[:, -1:, :, :]  ## last channel is vertical component
            # clip z to [-1, 1] after normalization
            z = torch.clamp(z, -1.0, 1.0)
            enc_polarity = self.encoder_polarity(z)

        enc1 = self.input_conv(x)
        enc2 = self.encoder12(enc1)
        enc3 = self.encoder23(enc2)
        enc4 = self.encoder34(enc3)
        enc5 = self.encoder45(enc4)

        if self.add_stft:
            nb, nc, nx, nt = x.shape
            sgram = x.clone()
            sgram = sgram.permute(0, 2, 1, 3).reshape(nb * nx, nc, nt)  # nb*nx, nc, nt
            sgram = self.stft(sgram)  # nb*nx, nc, nf, nt
            enc2_tf = self.encoder12_tf(sgram)  # nb*nx, nc, nf, nt
            enc3_tf = self.encoder23_tf(enc2_tf)
            enc4_tf = self.encoder34_tf(enc3_tf)
            enc5_tf = self.encoder45_tf(enc4_tf)

            enc2_tf = self.merge_freq2(enc2_tf)  # nb*nx, nc, nt
            enc2_tf = enc2_tf.view(nb, nx, *enc2_tf.shape[-2:]).permute(0, 2, 1, 3)  # nb, nc, nx, nt
            enc3_tf = self.merge_freq3(enc3_tf)
            enc3_tf = enc3_tf.view(nb, nx, *enc3_tf.shape[-2:]).permute(0, 2, 1, 3)
            enc4_tf = self.merge_freq4(enc4_tf)
            enc4_tf = enc4_tf.view(nb, nx, *enc4_tf.shape[-2:]).permute(0, 2, 1, 3)
            enc5_tf = self.merge_freq5(enc5_tf)
            enc5_tf = enc5_tf.view(nb, nx, *enc5_tf.shape[-2:]).permute(0, 2, 1, 3)

            enc2 = self.merge_branch2(enc2, enc2_tf)
            enc3 = self.merge_branch3(enc3, enc3_tf)
            enc4 = self.merge_branch4(enc4, enc4_tf)
            enc5 = self.merge_branch5(enc5, enc5_tf)

        if self.add_prompt:
            out_prompt = enc5.clone()
        else:
            out_prompt = None

        dec4 = self.upconv54(enc5)

        dec4 = torch.cat((dec4, enc4), dim=1)
        dec3 = self.decoder43(dec4)
        if self.add_event:
            out_event = self.output_event(dec3)
        else:
            out_event = None
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec2 = self.decoder32(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.decoder21(dec2)
        if self.add_polarity:
            dec_polarity = torch.cat((dec1, enc_polarity), dim=1)
            out_polarity = self.output_polarity(dec_polarity)
        else:
            out_polarity = None
        dec1 = torch.cat((dec1, enc1), dim=1)
        out_phase = self.output_conv(dec1)

        if self.output_upsample is not None:
            out_phase = self.output_upsample(out_phase)
            if self.add_polarity:
                out_polarity = self.output_upsample(out_polarity)
            if self.add_event:
                out_event = self.output_upsample(out_event)
            if self.add_prompt:
                out_prompt = self.output_upsample(out_prompt)

        result = {"phase": out_phase, "polarity": out_polarity, "event": out_event, "prompt": out_prompt}

        if self.add_stft:
            result["spectrogram"] = sgram

        return result

    @staticmethod
    def encoder_block(in_channels, out_channels, kernel_size=(1, 7), stride=(1, 4), padding=(0, 3), name=""):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm1", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm2", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    @staticmethod
    def decoder_block(
        in_channels, out_channels, kernel_size=(1, 7), stride=(1, 4), padding=(0, 3), upsample="conv_transpose", name=""
    ):
        layers = [
            (
                name + "_conv1",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                ),
            ),
            (name + "_norm1", nn.BatchNorm2d(num_features=in_channels // 2)),
            (name + "_relu1", nn.ReLU(inplace=True)),
        ]
        if upsample == "interpolate":
            layers.extend(
                [
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=in_channels // 2,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm2", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                    (name + "_upsample", nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)),
                ]
            )
        elif upsample == "conv_transpose":
            layers.extend(
                [
                    (
                        name + "_conv2",
                        nn.ConvTranspose2d(
                            in_channels=in_channels // 2,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            output_padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "_norm2", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )

        return nn.Sequential(OrderedDict(layers))
