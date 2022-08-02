import numpy as np
import time
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import tcod

np.random.seed(5)
tempType = "normal"
rainType = "normal"

def diamond_square(n, bias=False):
    # The array must be square with edge length 2**n + 1
    N = 2 ** n + 1

    arr = np.zeros((N, N))
    # f scales the random numbers at each stage of the algorithm
    f = 1.0
    if not bias:
        # Initialise the array with random numbers at its corners
        arr[0::N - 1, 0::N - 1] = np.random.uniform(-1, 1, (2, 2))
    else:
        arr[0] = np.random.uniform(-1.5, -.5, N)
        arr[N - 1] = np.random.uniform(.5, 1.5, N)

    side = N - 1
    nsquares = 1
    while side > 1:
        sideo2 = side // 2

        # Diamond step
        for ix in range(nsquares):
            for iy in range(nsquares):
                x0, x1, y0, y1 = ix * side, (ix + 1) * side, iy * side, (iy + 1) * side
                xc, yc = x0 + sideo2, y0 + sideo2
                # Set this pixel to the mean of its "diamond" neighbours plus
                # a random offset.
                arr[yc, xc] = (arr[y0, x0] + arr[y0, x1] + arr[y1, x0] + arr[y1, x1]) / 4
                arr[yc, xc] += f * np.random.uniform(-1, 1)

        # Square step: NB don't do this step until the pixels from the preceding
        # diamond step have been set.
        for iy in range(2 * nsquares + 1):
            yc = sideo2 * iy
            for ix in range(nsquares + 1):
                xc = side * ix + sideo2 * (1 - iy % 2)
                if not (0 <= xc < N and 0 <= yc < N):
                    continue
                tot, ntot = 0., 0
                # Set this pixel to the mean of its "square" neighbours plus
                # a random offset. At the edges, it has only three neighbours
                for (dx, dy) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    xs, ys = xc + dx * sideo2, yc + dy * sideo2
                    if not (0 <= xs < N and 0 <= ys < N):
                        continue
                    else:
                        tot += arr[ys, xs]
                        ntot += 1
                arr[yc, xc] += tot / ntot + f * np.random.uniform(-1, 1)
        side = sideo2
        nsquares *= 2
        f /= 2

    #now we need to normalize the maps
    #get the min value
    minValue = 0
    for y in arr:
        if min(y) < minValue:
            minValue = min(y)
    arr = arr + (minValue * -1)
    maxValue = 1
    for y in arr:
        if max(y) > maxValue:
            maxValue = max(y)
    arr = arr/maxValue
    #add the negative to all the numbers to get the lowest value to 0
    #divide all the numbers by the maximum value
    #now it's between 0 and 1

    return arr

mapSize = 8
N = 2**mapSize+1

st = time.time()
heightMap = diamond_square(mapSize)
dt = round(time.time() - st, 1)
print("generated heightMap\t\t{} sec".format(dt))

st = time.time()
rainMap = diamond_square(mapSize)
dt = round(time.time() - st, 1)
print("generated rainMap\t\t{} sec".format(dt))

st = time.time()
tempMap = diamond_square(mapSize)
dt = round(time.time() - st, 1)
print("generated tempMap\t\t{} sec".format(dt))

def normalize_map(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#adjuting temperature in a different way
#add or subtract 1/10th of the distance from 0 or 1 (depending on if you're making it hotter or colder
def heat_world(temp_map, h):
    #h is the heat intensity modifier
    for y in range(len(temp_map)):
        for x in range(len(temp_map)):
            temp_map[y][x] += (1 - temp_map[y][x])/h
    return temp_map

def cool_world(temp_map, h):
    #h is the heat intensity modifier
    for y in range(len(temp_map)):
        for x in range(len(temp_map)):
            temp_map[y][x] -= (temp_map[y][x])/h
    return temp_map

def wet_world(rain_map, h):
    #h is the heat intensity modifier
    for y in range(len(rain_map)):
        for x in range(len(rain_map)):
            rain_map[y][x] += (1 - rain_map[y][x])/h
    return rain_map

def dry_world(rain_map, h):
    #h is the heat intensity modifier
    for y in range(len(rain_map)):
        for x in range(len(rain_map)):
            rain_map[y][x] -= (rain_map[y][x])/h
    return rain_map

heightMap = normalize_map(heightMap)

# as the height increases the temperature decreases
#tempMap = tempMap - (np.sqrt(heightMap)/2)


# there are 8 levels for rain
# fuck with the rain to get what we want
st = time.time()
rainMap = normalize_map(rainMap)

#rainMap = dry_world(rainMap, -3)
#rainMap = wet_world(rainMap, -2)

if rainType == "dry":
    rainSplines = [0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.95]
elif rainType == "normal":
    #rainSplines = [0.05, 0.1, 0.3, 0.45, 0.6, 0.75, 0.9]
    rainSplines = [0.1, 0.15, 0.3, 0.45, 0.6, 0.8, 0.9]
elif rainType == "wet":
    rainSplines = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for y in range(len(rainMap)):
    for x in range(len(rainMap)):
        if 0 <= rainMap[y][x] < rainSplines[0]:
            rainMap[y][x] = 0
        elif rainSplines[0] <= rainMap[y][x] < rainSplines[1]:
            rainMap[y][x] = 1
        elif rainSplines[1] <= rainMap[y][x] < rainSplines[2]:
            rainMap[y][x] = 2
        elif rainSplines[2] <= rainMap[y][x] < rainSplines[3]:
            rainMap[y][x] = 3
        elif rainSplines[3] <= rainMap[y][x] < rainSplines[4]:
            rainMap[y][x] = 4
        elif rainSplines[4] <= rainMap[y][x] < rainSplines[5]:
            rainMap[y][x] = 5
        elif rainSplines[5] <= rainMap[y][x] < rainSplines[6]:
            rainMap[y][x] = 6
        elif rainSplines[6] <= rainMap[y][x] :
            rainMap[y][x] = 7
rainMap = rainMap.astype(int)
dt = round(time.time() - st, 1)
print("splined rainMap\t\t\t{} sec".format(dt))

# there are 7 for temperature
st = time.time()
tempMap = normalize_map(tempMap)

#tempMap = heat_world(tempMap, -2)
#tempMap = cool_world(tempMap, -3)

if tempType == "hot":
    tempSplines = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
elif tempType == "normal":
    #tempSplines = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    tempSplines = [0.1, 0.15, 0.2, 0.4, 0.8, 0.9]
elif tempType == "cold":
    tempSplines = [0.2, 0.3, 0.5, 0.7, 0.8, 0.95]
for y in range(len(tempMap)):
    for x in range(len(tempMap)):
        if 0 <= tempMap[y][x] < tempSplines[0]:
            tempMap[y][x] = 0
        elif tempSplines[0] <= tempMap[y][x] < tempSplines[1]:
            tempMap[y][x] = 1
        elif tempSplines[1] <= tempMap[y][x] < tempSplines[2]:
            tempMap[y][x] = 2
        elif tempSplines[2] <= tempMap[y][x] < tempSplines[3]:
            tempMap[y][x] = 3
        elif tempSplines[3] <= tempMap[y][x] < tempSplines[4]:
            tempMap[y][x] = 4
        elif tempSplines[4] <= tempMap[y][x] < tempSplines[5]:
            tempMap[y][x] = 5
        elif tempSplines[5] <= tempMap[y][x] :
            tempMap[y][x] = 6


tempMap = tempMap.astype(int)
dt = round(time.time() - st, 1)
print("splined tempMap\t\t\t{} sec".format(dt))

##############
#plt.imshow(heightMap, cmap='magma')
#plt.axis('off')
#plt.show()
#
#plt.imshow(rainMap, cmap='magma')
#plt.axis('off')
#plt.show()
#
#plt.imshow(tempMap, cmap='turbo')
#plt.axis('off')
#plt.show()
#
##############
st = time.time()
biomeGuide = [
    [1,  1,  1,  2,  2,  2,  2,  2],
    [1,  1,  1,  3,  3,  4,  4,  5],
    [6,  6,  7,  8,  8,  8,  8,  5],
    [6,  6,  7,  9,  9, 10, 11, 12],
    [13, 14, 15, 16, 17, 10, 11, 12],
    [13, 14, 15, 16, 18, 19, 20, 21],
    [13, 14, 22, 23, 24, 25, 25, 26]
]
#biome mapping
# 0   ocean
# 1   polar desert
# 2   ice cap
# 3   tundra
# 4   wet tundra
# 5   polar wetlands
# 6   cool desert
# 7   step
# 8   boreal forest
# 9   temperate woodlands
# 10  temperate forest
# 11  temperate wet forest
# 12  temperate wetlands
# 13  extreme desert
# 14  desert
# 15  subtropical scrub
# 16  subtropical wetlands
# 17  mediterranean
# 18  subtropical dry forest
# 19  subtropical forest
# 20  subtropical wet forest
# 21  subtropical wetlands
# 22  tropical scrub
# 23  tropical woodlands
# 24  tropical dry forest
# 25  tropical wet forest
# 26  tropical wetlands

seaLevel = np.median(heightMap)

biomeMap = np.zeros((N, N))
for y in range(len(biomeMap)):
    for x in range(len(biomeMap)):
        if heightMap[y][x] < seaLevel:
            biomeMap[y][x] = 0
        else:
            biomeMap[y][x] = biomeGuide[tempMap[y][x]][rainMap[y][x]]
dt = round(time.time() - st, 1)
print("generated biomeMap\t\t{} sec".format(dt))

np.savetxt("map_m3.csv", biomeMap, delimiter=",")


# generate cities
st = time.time()
cityMap = np.zeros((N,N))
numCities = N/2
while numCities > 0:
    rY = np.random.randint(0,N)
    rX = np.random.randint(0,N)
    if heightMap[rY][rX] > seaLevel:
        if not biomeMap[rY][rX] in [1, 2, 13]:
            cityMap[rY][rX] = 1
            numCities -= 1

# adjust the size
# pick a random city to add 1 pop to
# add that city back into the pool again
# get the list of cities
cityList = []
for y in range(len(cityMap)):
    for x in range(len(cityMap)):
        if cityMap[y][x] > 0:
            cityList.append([y, x])


worldPop = len(cityList)
for i in range(worldPop):
    cityNum = np.random.randint(0,len(cityList))
    cityToAdjust = cityList[cityNum]

    # this makes it zipfian
    # adjust based on temp and rainfall
    if biomeMap[cityToAdjust[0]][cityToAdjust[1]] in [0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 22, 26, 25, 21]:
        if np.random.uniform(0,1) < 1/4:
            cityMap[cityToAdjust[0]][cityToAdjust[1]] += 1
            cityList.append(cityToAdjust)
    else:
        cityMap[cityToAdjust[0]][cityToAdjust[1]] += 1
        cityList.append(cityToAdjust)

dt = round(time.time() - st, 1)
print("generated cityMap\t\t{} sec".format(dt))

np.savetxt("citymap_m3.csv", cityMap, delimiter=",")

st = time.time()
# making roads
numRoads = 100
# make a heatmap that shows the distance to the nearest city tile
# have a pathfinding thing pick two random cities and connect them via a walk
# remove any that are on water

# set a map to 100
cost = np.ones((N,N))
cost *= 100

# subtract from it the population of the city
cost = cost - cityMap

# blur it
cost = gaussian_filter(cost, sigma=4)
cost = normalize_map(cost)
cost *= 100

#add the mountains
for y in range(len(heightMap)):
    for x in range(len(heightMap)):
        if not heightMap[y][x] <= seaLevel:
            cost[y][x] += heightMap[y][x]*100

# subtract the ocean
for y in range(len(cost)):
    for x in range(len(cost)):
        if heightMap[y][x] <= seaLevel:
            cost[y][x] = 0

# convert floats to ints
cost = cost.astype(int)

#add some randomness
for y in range(len(cost)):
    for x in range(len(cost)):
        if cost[y][x] > 0:
            cost[y][x] += np.random.randint(0,50)

#add a penalty for certain biomes
for y in range(len(cost)):
    for x in range(len(cost)):
        if biomeMap[y][x] in [0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 22, 26, 25, 21]:
            cost[y][x] *= 5

graph = tcod.path.CustomGraph((N, N))
CARDINAL = [
    [141, 100, 141],
    [100, 0, 100],
    [141, 100, 141],
]

roadMap = np.zeros((N, N))

#add from random points on the edge to make the roads not be self contained to the map
#we want at least 2
numBorderRoads = mapSize - 3
for i in range(numBorderRoads):
    #pick a cardinal direction
    border = np.random.randint(0,4)
    if border == 0:
        cityList.append([0, np.random.randint(0,N)])
    elif border == 1:
        cityList.append([N - 1, np.random.randint(0, N)])
    elif border == 2:
        cityList.append([np.random.randint(0, N), 0])
    elif border == 3:
        cityList.append([np.random.randint(0, N), N-1])

# randomize the city list
#cityList = cityList * 2
np.random.shuffle(cityList)
startCity = cityList[0]

from scipy.spatial import distance
def closest_node_loop(node, nodes, n):
    for i in range(n):
        closest_index = distance.cdist([node], nodes).argmin()
        if i < n-1:
            del nodes[closest_index]
    return nodes[closest_index]

cityIndexes = np.random.randint(0, len(cityList), 2)

result = []
for i in cityList:
    if i not in result:
        result.append(i)

cityList = result.copy()

#connect each city to its nth closest neighbor
for n in range(1,5):
    for j in range(len(cityList)):
        startCity = cityList[j]
        #cityIndexes = np.random.randint(0, len(cityList), 2)
        #remove the city from the cityList
        cityListSubset = cityList.copy()
        del cityListSubset[j]
        endCity = closest_node_loop(startCity, cityListSubset,n)
        # get the path
        graph.add_edges(edge_map=CARDINAL, cost=cost)
        pf = tcod.path.Pathfinder(graph)
        pf.add_root((startCity[0], startCity[1]))
        road = pf.path_to((endCity[0], endCity[1]))
        #make the road map
        for i in road:
            roadMap[i[0]][i[1]] += 1
            #adjust the cost based on the roadMap so the roads converge
            cost[i[0]][i[1]] /= 3
        #startCity = endCity


dt = round(time.time() - st, 1)
print("generated roadMap\t\t{} sec".format(dt))

np.savetxt("roadmap_m3.csv", roadMap, delimiter=",")

# we're going to generate rivers
# this will take the most computer time
# make a riverMap of zeros

print("done")
