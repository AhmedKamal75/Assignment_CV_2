import cv2
import numpy as np
import os


def are_equal(img1, img2):
    return (img1.shape == img2.shape) and not (np.bitwise_xor(img1, img2).any())


def load_imgs(img_paths):
    imgs = []
    for path in img_paths:
        img = cv2.imread(path)
        imgs.append(img)

    return imgs


# matrix to go from src ==> to dst
def get_matrix(src, dst):
    # generate key points and descriptors
    sift = cv2.SIFT_create()
    key_points1, descriptor1 = sift.detectAndCompute(src, None)
    key_points2, descriptor2 = sift.detectAndCompute(dst, None)

    # get matched points
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) >= 50:
        # filter the needed points (found through matcher) from the key points provided by sift, meaning find
        # correspondences
        points1 = np.float32([key_points1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32([key_points2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

        # find matrix 
        matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    else:
        return None, None, None

    # img_features = cv2.drawMatches(src, key_points1, dst, key_points2,good_matches, None, flags=2)
    # cv2.imwrite("output.jpg",img_features)

    return matrix
    #
    # # print(new_shape)
    # new_mask = np.ones(shape=(new_shape[1], new_shape[0]), dtype=np.uint8) * 255
    # cv2.fillPoly(new_mask, [np.int32(nc)], (0, 0, 0))
    # new_mask = cv2.bitwise_and(new_mask, new_mask, wi)
    # # print(type(new_shape))
    # # test_img = cv2.polylines(new_mask, [np.int32(nc)], True, (255, 0, 255), 3)
    # cv2.imshow("test", new_mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cv2.imwrite("test.jpg", new_mask)

#
# def stitch_all(imgs, output_path):
#     result = None
#     l = len(imgs)
#     if l < 0:
#         result = None
#     if l == 1:
#         result = imgs[0]
#     if l > 1:
#         src = imgs[0]
#         for i in range(1, l):
#             dst = imgs[i]
#             stitched_image = stitch(src, dst)
#             src = stitched_image
#
#     cv2.imwrite(output_path, stitched_image)


# src is the one to the left --> will be warped , dst is the one that is on the right --> will be the new origin
def stitch(src, dst, output_path):
    matrix = get_matrix(src=src, dst=dst)
    output = cv2.warpPerspective(src, matrix, (src.shape[1] + dst.shape[1], dst.shape[0]))
    output[0:dst.shape[0], 0:dst.shape[1]] = dst
    cv2.imwrite(output_path, output)
    return output


if __name__ == '__main__':
    imgs_paths = []
    for i in range(1, 3, 1):
        imgs_paths.append(os.path.join("assignment_2_materials", "part_2_imgs", "pano_image" + str(i) + ".jpg"))
        # imgs_paths.append(os.path.join("assignment_2_materials","part_2_imgs", "test_3" ,str(i) + ".JPG"))

    imgs = load_imgs(imgs_paths)

    # stitch(imgs[1],imgs[0],os.path.join("output","part_2_output_2_test_2.jpg"))
    stitch(imgs[0], imgs[1], os.path.join("output", "part_2_output_2_test_1.jpg"))
