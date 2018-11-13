from importlib import import_module
from . import (Srcnn, Espcn, Dcscn, DnCnn, Vdsr, Drcn, Drrn, LapSrn, MemNet,
               Edsr, Idn, Rdn, Dbpn, Carn, Rcan, SrGan, Vespcn)

__all__ = ['get_model', 'list_supported_models']

alias = {
    'srcnn': Srcnn.SRCNN,
    'espcn': Espcn.ESPCN,
    'vdsr': Vdsr.VDSR,
    'drcn': Drcn.DRCN,
    'dncnn': DnCnn.DnCNN,
    'idn': Idn.InformationDistillationNetwork,
    'rdn': Rdn.ResidualDenseNetwork,
    'dcscn': Dcscn.DCSCN,
    'lapsrn': LapSrn.LapSRN,
    'drrn': Drrn.DRRN,
    'memnet': MemNet.MEMNET,
    'dbpn': Dbpn.DBPN,
    'edsr': Edsr.EDSR,
    'srgan': SrGan.SRGAN,
    'carn': Carn.CARN,
    'rcan': Rcan.RCAN,
    'vespcn': Vespcn.VESPCN,
}

# module in development
exp = {
    # name: (package, class)
    'lapgan': ('Exp.LapGAN', 'LapGAN'),
    'lapres': ('Exp.LapRes', 'LapRes'),
    'flownets': ('Exp.FlowNet', 'FlowNetS'),
    'frvsr': ('Exp.Frvsr', 'FRVSR'),
    'evsr': ('Exp.Flow', 'EVSR'),
    'duf': ('Exp.Duf', 'DUF'),
    'sgan': ('Exp.Gan', 'StdGAN'),
    'gangp': ('Exp.Gan', 'GANGP'),
    'lsgan': ('Exp.Gan', 'LSGAN'),
    'wgan': ('Exp.Gan', 'WGAN'),
    'wgangp': ('Exp.Gan', 'WGANGP'),
    'rgan': ('Exp.Gan', 'RGAN'),
    'ragan': ('Exp.Gan', 'RaGAN'),
    'rlsgan': ('Exp.Gan', 'RLSGAN'),
    'ralsgan': ('Exp.Gan', 'RaLSGAN'),
    'srcgan': ('Exp.SRcGan', 'SRCGAN'),
    'exp': ('Exp.Exp', 'Exp'),
}

for module in exp:
    try:
        m = import_module(exp[module][0])
        alias.update({
            module: m.__dict__[exp[module][1]]
        })
    except ImportError:
        print('Warning: missing', module)


def get_model(name):
    return alias[name]


def list_supported_models():
    return alias.keys()
