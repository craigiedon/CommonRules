# Instructions for Running on the EDDIE Cluster

### Creating a Conda Environment

Listing conda envs:

    conda env list

Installing packages to a conda env from a pip requirements?

    conda install pip
    pip install


Theres some note about making sure pytorch is installed with the cuda versions


Remember: Configuration options set here (including some stuff about caches...)
https://www.wiki.ed.ac.uk/display/ResearchServices/Anaconda

    module load anaconda
    module load cuda
    conda activate ams-runner

Remember - you must allocate appropriate vmem too! (8G?)

