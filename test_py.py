import pymc3
import bioassay_model

with bioassay_model:

    # Draw wamples
    trace = pymc3.sample(1000, njobs=2)
    # Plot two parameters
    pymc3.forestplot(trace, varnames=['alpha', 'beta'])
