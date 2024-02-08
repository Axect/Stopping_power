import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# Import parquet file
df = pd.read_parquet('./stp_pow.parquet')

# Prepare Data to Plot
x = df['E_k']
y = df['stp_pow']

# Plot params
pparam = dict(
    xlabel = r'$E_{\rm kin}$ (MeV)',
    ylabel = r'Linear stopping power (MeV/cm)',
    xscale = 'log',
    yscale = 'log',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.plot(x, y, label=r'Total stopping power of $e^+$')
    ax.legend()
    fig.savefig('stp_pow_plot.png', dpi=600, bbox_inches='tight')
