import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Basic SIREN layers.
class SineAffine(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            freq: float = 30.0,
            start: bool = False,
            use_shift: bool = False,
            shift=None,
    ):
        super(SineAffine, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.freq = freq
        self.start = start
        self.use_shift = use_shift
        if use_shift:
            assert shift.size(0) == out_features
            self.shift = shift

        # Affine transformation.
        self.affine = nn.Linear(in_features, out_features, bias=True)
        self._init_affine()

    def _init_affine(self):
        # Initialize the parameters.
        b = 1 / self.in_features if self.start else math.sqrt(6 / self.in_features) / self.freq
        nn.init.uniform_(self.affine.weight, -b, b)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x):
        if self.use_shift:
            out = self.affine(x) + self.shift.unsqueeze(0)
            out = torch.sin(self.freq * out)
        else:
            out = self.affine(x)
            out = torch.sin(self.freq * out)
        return out


class SIREN(nn.Module):
    def __init__(
            self,
            hidden_features: int,
            num_layers: int,
            freq: float = 30.0,
            use_shift: bool = False,
    ):
        super(SIREN, self).__init__()
        # Set parameters.
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.freq = freq
        self.use_shift = use_shift

        # Construct the layers.
        self.net = self._make_layers()
        self.hidden2rgb = nn.Linear(hidden_features, 1, bias=True)
        b = math.sqrt(6 / hidden_features) / freq
        nn.init.uniform_(self.hidden2rgb.weight, -b, b)
        nn.init.zeros_(self.hidden2rgb.bias)

    def _make_layers(self):
        assert self.num_layers > 0
        layers = []
        for i in range(self.num_layers):
            in_features = 2 if i == 0 else self.hidden_features
            if self.use_shift:
                layers.append(
                    SineAffine(
                        in_features, self.hidden_features, self.freq, start=(i == 0),
                        use_shift=True, shift=torch.zeros(self.hidden_features, )
                    )
                )
            else:
                layers.append(
                    SineAffine(
                        in_features, self.hidden_features, self.freq, start=(i == 0), use_shift=False,
                    )
                )
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        out = self.hidden2rgb(out)
        return out


class ModulatedSIREN(nn.Module):
    def __init__(
            self,
            height: int,
            width: int,
            hidden_features: int,
            num_layers: int,
            modul_features: int,
            freq: float = 30.0,
            device='cuda'
    ):
        super(ModulatedSIREN, self).__init__()

 
        self.height = height
        self.width = width
        x, y = torch.meshgrid(torch.arange(height), torch.arange(width))
        x = x.float().view(-1).unsqueeze(0).to(device)
        y = y.float().view(-1).unsqueeze(0).to(device)
        self.meshgrid = torch.cat((x, y), dim=0).T

      
        self.siren = SIREN(
            hidden_features=hidden_features,
            num_layers=num_layers,
            freq=freq,
            use_shift=True,
        )

       
        self.modul_features = modul_features
        self.modul = nn.Linear(modul_features, hidden_features * num_layers)

    def assign_shift(self, shift):
        
        hidden_features = self.siren.hidden_features
        assert shift.size(0) == hidden_features * self.siren.num_layers
        i = 0
        for layer in self.siren.net:
            layer.shift = shift[i * hidden_features: (i + 1) * hidden_features]
            i += 1

    def forward(self, phi):
       
        shift = self.modul(phi)
        self.assign_shift(shift=shift)
        coord = self.meshgrid.clone()
        out = self.siren(coord)
        return out

class ModulatedSIREN_IM(nn.Module):
    def __init__(
            self,
            height: int,
            width: int,
            hidden_features: int,
            num_layers: int,
            modul_features: int,
            freq: float = 30.0,
            device = 'cuda'
    ):
       
        super(ModulatedSIREN_IM, self).__init__()

        # Generate a mesh grid.
        self.height = height
        self.width = width
        x, y = torch.meshgrid(torch.arange(height), torch.arange(width))
        x = x.float().view(-1).unsqueeze(0).to(device)
        y = y.float().view(-1).unsqueeze(0).to(device)
        self.meshgrid = torch.cat((x, y), dim=0).T

        # Construct the layers.
        self.siren = SIREN(
            hidden_features=hidden_features,
            num_layers=num_layers,
            freq=freq,
            use_shift=True,
        )
        
        self.siren.hidden2rgb = nn.Linear(hidden_features, 3, bias=True)

        # Modulation.
        self.modul_features = modul_features
        self.modul = nn.Linear(modul_features, hidden_features * num_layers)

    def assign_shift(self, shift):
        
        hidden_features = self.siren.hidden_features
        assert shift.size(0) == hidden_features * self.siren.num_layers
        i = 0
        for layer in self.siren.net:
            layer.shift = shift[i * hidden_features: (i + 1) * hidden_features]
            i += 1

    def forward(self, phi):
       
        shift = self.modul(phi)
        self.assign_shift(shift=shift)
        coord = self.meshgrid.clone()
        out = self.siren(coord)
        return out
