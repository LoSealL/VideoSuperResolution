# Export model objects
from VSR.Models import *
from importlib import import_module

__all__ = {
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
}

# module in development
__exp__ = {
    # name: (package, class)
    'lapgan': ('Exp.LapGAN', 'LapGAN'),
    'pwc': ('Exp.PWC', 'PWC'),
    'flownets': ('Exp.FlowNet', 'FlowNetS'),
    'vespcn': ('Exp.Vespcn', 'VESPCN'),
}

for module in __exp__:
    try:
        m = import_module(__exp__[module][0])
        __all__.update({
            module: m.__dict__[__exp__[module][1]]
        })
    except ImportError:
        print('Warning: missing', module)


def get_model(name):
    return __all__[name]


def list_supported_models():
    return __all__.keys()
