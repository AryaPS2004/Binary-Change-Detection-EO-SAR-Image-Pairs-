import rasterio

path = r"./datasets/train/pre-event/scene_01_000001_building_damage.tif"

with rasterio.open(path) as src:
    print("Number of bands:", src.count)