
#from roboflow import Roboflow

rf = Roboflow(api_key="aN6EfdnektrGuY4cg1rO")

# get a workspace
workspace = rf.workspace("caipirinia")

# Upload data set to a new/existing project
workspace.upload_dataset(
    "./datasets/", # This is your dataset path
    "popoche", # This will either create or get a dataset with the given ID
    num_workers=10,
    project_license="MIT",
    project_type="object-detection",
    batch_name=None,
    num_retries=0
)




# !pip install roboflow
#
# from roboflow import Roboflow
# rf = Roboflow(api_key="aN6EfdnektrGuY4cg1rO")
# project = rf.workspace("caipirinia").project("midnight-nleml")
# version = project.version(1)
# dataset = version.download("createml")
