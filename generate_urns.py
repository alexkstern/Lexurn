import torch

def generate_urns(n_tasks=10,n_colors=4,alpha=1.0,device="cpu",seed=None):

    """
    Generate D urns (tasks), each as a categorical distribution over `dim` colors,
    using Dirichlet sampling in PyTorch.
    #takes D number of tasks
    #Takes dim as the number of colors in an urn
    alpha=1 Uniform over the simplex (neutral) hence every color has equal chance of being some number between 0 and 1
    #takes device='cpu'as default
    """

    if seed is not None:
        torch.manual_seed(seed)
    

    alpha_vec = torch.full(size=(n_colors,), fill_value=float(alpha), device=device) 
    urns = torch.distributions.Dirichlet(alpha_vec).sample((n_tasks,)) #Dirichlet distribution is a prob distribution that generates prob distributions
    return urns  # shape: (D, dim)


"""
Plan: class that takes in my context length, the number of colors, the number of tasks, the size of dataset which is equivalent 
to the number of steps
(can get from the config file in the configs/dummy.config)
config looks like this:
"
[dataset]
context=8
d_tasks=32
n_colors=4

"

then it can create the dataset, that i will then feed into a data loader,

perhaps the dataset can just be a massive tensor ultimately 


"""

if __name__=="main":

    D=20
    dim=4
    example=generate_urns(n_tasks=D,n_colors=dim)
    print(example)