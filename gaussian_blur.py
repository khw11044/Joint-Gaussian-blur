import numpy as np
import matplotlib.pyplot as plt
import cv2 

def generate_heatmaps(pose, size, sigma, joint_weights=np.array([0.93, 0.99, 0.34, 0.78, 1., 1., 1., 1., 1., 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5])):
    sigma = int(sigma * size / 128)
    heatmaps = np.zeros((pose.shape[0], size + 2 * sigma, size + 2 * sigma))
    weights = joint_weights 
    win_size = 2 * sigma + 1

    x, y = np.meshgrid(np.linspace(-sigma, sigma, num=win_size, endpoint=True), np.linspace(-sigma, sigma, num=win_size, endpoint=True))
    dst = np.sqrt(x * x + y * y)
    mu = 0.000
    gauss = np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2)))

    for i, [X, Y] in enumerate(pose):
        X, Y = int(X), int(Y)   
        if X <= 0 or X >= size or Y <= 0 or Y >= size:
            weights[i] = 0
            continue

        heatmaps[i, Y: Y + win_size, X: X + win_size] = gauss * weights[i]

    heatmaps = heatmaps[:, sigma:-sigma, sigma:-sigma]

    return heatmaps, weights

image_size = (1000,1000)
dummy_img = np.zeros((image_size[0], image_size[1], 3), np.uint8)

poses = np.array([[500.     , 500.     ],
                [567.63983, 507.64615],
                [515.2447 , 712.59186],
                [580.5729 , 939.89404],
                [431.19388, 492.25705],
                [456.67462, 728.3733 ],
                [484.2139 , 967.63226],
                [518.18976, 241.09186],
                [511.97144, 187.62656],
                [529.9296 , 135.03442],
                [451.29324, 289.04358],
                [391.96268, 427.6672 ],
                [349.70953, 519.674  ],
                [585.3179 , 280.95026],
                [615.4844 , 421.6824 ],
                [603.48834, 528.4487 ]])

heatmaps, weights = generate_heatmaps(pose=poses, size=image_size[0], sigma=2)

img0=dummy_img.copy()
fig = plt.figure(figsize=(12,6))

ax0 = fig.add_subplot(1,3,1)
for j in range(len(poses)):
    joint=tuple(poses[j].astype(int))
    cv2.circle(img0, joint, 8, (255,0,255), -1)

# plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
plt.imshow(img0)


ax1 = fig.add_subplot(1,3,2)
plt.imshow(heatmaps[0])

print(np.max(heatmaps[0]))
print(np.max(heatmaps[1]))
print(np.max(heatmaps[2]))
print(np.max(heatmaps[3]))

ax1 = fig.add_subplot(1,3,3)
plt.imshow(np.sum(heatmaps,axis=0))

plt.savefig('img')
plt.show()