import xir
import sys

xmodel = sys.argv[1] if len(sys.argv) > 1 else "croprow_yolov8.xmodel"

graph = xir.Graph.deserialize(xmodel)
root = graph.get_root_subgraph()

def walk(sg, level=0):
    indent = "  " * level
    name = sg.get_name()
    attrs = []
    for a in ["device", "runner", "reg_id_to_context_type"]:
        if sg.has_attr(a):
            attrs.append(f"{a}={sg.get_attr(a)}")
    print(f"{indent}- {name} | " + ", ".join(attrs))

    for child in sg.toposort_child_subgraph():
        walk(child, level + 1)

print("Inspect:", xmodel)
walk(root)
