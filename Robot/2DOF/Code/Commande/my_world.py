from klampt import WorldModel
from klampt import vis

w = WorldModel()
if not w.readFile("klampt_data/world.xml"):
    raise RuntimeError("Couldn't read the world file")
vis.add("world",w)
vis.run()