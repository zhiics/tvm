import sys
import numpy as np
import tvm
from tvm import relay
from argparse import ArgumentParser
#from tvm.relay import vmobj as _obj

import mxnet
from mxnet.gluon.model_zoo import vision

target = "llvm -mcpu=skylake-avx512"
ctx = tvm.cpu()
# target = "cuda"
# ctx = tvm.gpu()

# model = "mobilenet0.75"
model = "resnet18_v1"
# model = "resnet18_v2"
# model = "resnet34_v1"
# model = "resnet50_v1"
# model = "squeezenet1.0"
# model = "vgg16"
# model = "densenet121"
# model = "inceptionv3"
# model = "maskrcnn"

parser = ArgumentParser()

parser.add_argument("-r", "--runtime", choices=["graph", "vm"],
                    default="graph", help="runtime, graph vs vm")
parser.add_argument("--profile", action="store_true", help="Profile the performance")
args = parser.parse_args()
rt = args.runtime

dshape=(1, 3, 224, 224)
data = np.random.random(dshape).astype('float32')

def load_mx(model):
    model = vision.get_model(model, pretrained=True)
    mod, params = tvm.relay.frontend.from_mxnet(model,
                                                shape={'data':dshape})
    return mod, params

mod, params = load_mx(model)

def create_profiler():
    global args, ctx, mod, params, ipnuts, token_types, valid_length
    if rt == "graph":
        graph, lib, cparams = relay.build(mod["main"], target, params=params)
        from tvm.contrib.debugger import debug_runtime
        prof = debug_runtime.create(graph, lib, ctx)
        prof.set_input(**cparams)
        prof.set_input("data", data)
    else:
        from tvm.runtime import profiler_vm
        exe = relay.vm.compile(mod, target, params=params)
        prof = profiler_vm.VirtualMachineProfiler(exe)
        prof.init(ctx)
        prof.set_input("main", data)
    return prof

def graph_runtime(mod, params):
    print("measure graph_runtime....")
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)

    output = f"lib_{args.runtime}_{model}.txt"
    with open(output, 'w') as f:
        f.write(lib.get_source())

    m = tvm.contrib.graph_runtime.create(graph, lib, ctx)
    m.set_input("data", data)
    m.set_input(**params)
    m.run()
    warm = m.module.time_evaluator("run", ctx, number=1, repeat=10)
    warm()
    ftimer = m.module.time_evaluator("run", ctx, number=2, repeat=100)
    prof_res = np.array(ftimer().results) * 1000
    print("Mean graph_runtime inference time (std dev): %.2f ms (%.2f ms)"
          %(np.mean(prof_res), np.std(prof_res)))

def vm(mod, params):

    with relay.build_config(opt_level=3):
        exe = tvm.relay.vm.compile(mod, target, params=params)
        output = f"lib_{args.runtime}_{model}.txt"
        with open(output, 'w') as f:
            f.write(exe.lib.get_source())
        vm = tvm.runtime.vm.VirtualMachine(exe)
        vm.init(ctx)

    vm.set_input("main", data=data)
    vm.run()
    vm.reset()

    for _ in range(10):
        vm.run()

    vm.print_time()

    warm = vm.mod.time_evaluator("invoke", ctx, number=1, repeat=100)
    prof_res = np.array(warm("main").results) * 1000

    print("Mean vm inference time (std dev): %.2f ms (%.2f ms)"
          % (np.mean(prof_res), np.std(prof_res)))

if args.profile:
    with tvm.transform.PassContext(opt_level=3):
        prof = create_profiler()

    if rt == "vm":
        for _ in range(3):
            prof.run()
        prof.reset()

    prof.run()
    if rt == "vm":
        print(prof.get_stat(True))

    # for _ in range(10):
    #     prof.run()

    # output = f"prof_{args.runtime}_{model}.txt"

    # with open(output, 'w') as f:
    #     if rt == "graph":
    #         f.write(prof.debug_datum.get_debug_result(False))
    #     else:
    #         f.write(prof.get_stat(False))
else:
    if rt == "graph":
        graph_runtime(mod, params)
    else:
        vm(mod, params)
