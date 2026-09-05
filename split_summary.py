import json, sys

TARGET_TASKS = dict(
    atomic_seen=[
        "CloseBlenderLid","CloseFridge","CloseToasterOvenDoor","CoffeeSetupMug",
        "NavigateKitchen","OpenCabinet","OpenDrawer","OpenStandMixerHead",
        "PickPlaceCounterToCabinet","PickPlaceCounterToStove","PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter","PickPlaceToasterToCounter","SlideDishwasherRack",
        "TurnOffStove","TurnOnElectricKettle","TurnOnMicrowave","TurnOnSinkFaucet",
    ],
    composite_seen=[
        "DeliverStraw","GetToastedBread","KettleBoiling","LoadDishwasher",
        "PackIdenticalLunches","PreSoakPan","PrepareCoffee","RinseSinkBasin",
        "ScrubCuttingBoard","SearingMeat","SetUpCuttingStation","StackBowlsCabinet",
        "SteamInMicrowave","StirVegetables","StoreLeftoversInBowl","WashLettuce",
    ],
    composite_unseen=[
        "ArrangeBreadBasket","ArrangeTea","BreadSelection","CategorizeCondiments",
        "CuttingToolSelection","GarnishPancake","GatherTableware","HeatKebabSandwich",
        "MakeIceLemonade","PanTransfer","PortionHotDogs","RecycleBottlesByType",
        "SeparateFreezerRack","WaffleReheat","WashFruitColander","WeighIngredients",
    ],
)

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

tasks = data["tasks"]
results = {}
all_successes = 0
all_episodes = 0
for split, names in TARGET_TASKS.items():
    succ = sum(tasks[n]["successes"] for n in names)
    eps = sum(tasks[n]["num_episodes"] for n in names)
    results[split] = succ / eps
    all_successes += succ
    all_episodes += eps

print(f"file: {path}")
for split in ["atomic_seen", "composite_seen", "composite_unseen"]:
    print(f"  {split}: {results[split]*100:.2f}%  ({sum(tasks[n]['successes'] for n in TARGET_TASKS[split])}/{sum(tasks[n]['num_episodes'] for n in TARGET_TASKS[split])})")
print(f"  overall: {all_successes/all_episodes*100:.2f}%  ({all_successes}/{all_episodes})")
print(f"  (top-level episode_success_rate field: {data.get('episode_success_rate')})")
