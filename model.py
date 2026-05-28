# flowlib/model.py

from nflows import transforms, distributions, flows

#from flowlib.uniform import NflowsBoxUniformBase #This does not work. 
import torch

def build_flow(ndim, num_layers, hidden_features=64, num_blocks=2, use_lu_linear=False):

    transform_list = []

    for _ in range(num_layers):
        transform_list.append(
            transforms.MaskedAffineAutoregressiveTransform( #implements a MAF from (from https://arxiv.org/abs/1705.07057) - this is the invertible transform/transformer of the flow, each layer of which contains a MADE conditioner (as per the description in Prince). In the nflows package, each MAAF layer consists of num_blocks residual blocks, each consisting of two linear layers, plus an initial and final layer; see lab notebook. 
                features=ndim,
                hidden_features=hidden_features,
                num_blocks=num_blocks,
                dropout_probability=0.1
            )
        )

        if use_lu_linear: #adding lu_linear layers could improve modelling of correlations, but with MAAF layers used already they do not seem to add much.
            transform_list.append(
                transforms.LULinear(features=ndim)
            )

        transform_list.append( #The random permutation changes the ordering used by the mask between layers, so no ordering dominates. This is important as "autoregressive models can be sensitive to the order of the variables."
            transforms.RandomPermutation(features=ndim)
        )

    transform = transforms.CompositeTransform(transform_list)
    base_dist = distributions.StandardNormal(shape=[ndim]) #Defines a prior distribution (multivariate normal; changing this is infeasible without significant code changes elsewhere):
    #base_dist = NflowsBoxUniformBase( #tried this again 20th May, still cannot get it to work. I get nans. 
        #low=-8*torch.ones(ndim),
        #high=8*torch.ones(ndim)
    #)
    flow = flows.Flow(transform, base_dist)

    #flow = flows.Flow(transform, base_dist).to(device)

    return flow


def build_flowspline(ndim, num_layers, hidden_features=64, num_blocks=2, num_bins=2):

    transform_list = []

    for _ in range(num_layers):
        transform_list.append(transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=ndim,hidden_features = hidden_features, num_blocks=num_blocks, num_bins=num_bins, tails='linear', tail_bound=4)) #implements a spline flow
        #transform_list.append(transforms.LULinear(features = ndim)) #appending LULinear layer to see effect of correlations. 
        transform_list.append(transforms.RandomPermutation(features=ndim))
        
    transform = transforms.CompositeTransform(transform_list)
    base_dist = distributions.StandardNormal(shape=[ndim])
    flow = flows.Flow(transform, base_dist)

    return flow