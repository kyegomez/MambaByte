[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# MambaByte
Implementation of MambaByte in "MambaByte: Token-free Selective State Space Model" in Pytorch and Zeta. Note this will be a higher performance implementation of Mamba with parallel scan 


## Installation

```bash
pip install mambabyte
```

# Usage
```python
import torch 
from mambabyte import MambaConfig, Mamba

x = torch.randn(2, 3, 4)
config = MambaConfig(
    dim = 4,
    depth = 3,
    dt_rank = 2,
    d_state = 2,
    expand_factor = 2,
    d_conv = 3,
    dt_min = 0.001,
    dt_max = 0.1,
    dt_init = "random",
    dt_scale = 1.0,
    bias = False,
    conv_bias = True,
    pscan = True
)

model = Mamba(config)

out = model(x)

print(out)

```


# License
MIT
