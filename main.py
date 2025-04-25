WAIT = 30
WARMUP = 30
ACTIVE = 40
BS = 32

def monitor_torch_cuda(expname):
    import torch, torchvision, pathlib
    from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(expname),
        schedule=torch.profiler.schedule(
            wait=WAIT,
            warmup=WARMUP,
            active=ACTIVE,
            repeat=1,
        ),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True,
        acc_events=True,
    ) as profiler:
        model = torchvision.models.resnet18().to("cuda")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
        data = torchvision.datasets.CIFAR10(
            root="./.cifar10",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((224, 224)),
            ]),
        )
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=BS,
            shuffle=False,
            num_workers=0,
        )
        for i, (x, y) in enumerate(loader):
            with record_function("get_input"):
                x = x.to("cuda")
                y = y.to("cuda")
            with record_function("forward"):
                pred = model(x)
                loss = criterion(pred, y)
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer_step"):
                optimizer.zero_grad()
                optimizer.step()
            profiler.step()
            print(i)
            if i == WAIT + WARMUP + ACTIVE:
                break
    with pathlib.Path(f"{expname}.info").open("w") as f:
        f.write(profiler.key_averages().table(row_limit=20, sort_by="cuda_time_total"))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--expname", type=str, required=True)
    args = parser.parse_args()
    monitor_torch_cuda(args.expname)