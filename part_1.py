import numpy as np
import cv2
import os


def get_matches(descriptor_1, descriptor_2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)

    # filter the acceptable matches
    acceptable_matches = []
    for i, j in matches:
        if i.distance < 0.75 * j.distance:
            acceptable_matches.append(i)

    # acceptable_matches --> item(s) , item --> (item.queryIdx,item.trainIdx) , those are indexes in the key points
    # associated with descriptors 1 and 2

    return acceptable_matches


# kp1 --> query, kp2 --> train
def get_matrix_and_destination(matches, kp1, kp2, src):
    # extract the points from that targeted frame and the target frame, you also know that they matches.
    # query --> train
    source_points = np.float32([kp1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    destination_points = np.float32([kp2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5)
    return matrix, cv2.perspectiveTransform(src, matrix)


def merge_frames(frame_on, warped_frame_from, destination):
    # final image to write upon
    augmented_image = frame_on.copy()
    new_mask = np.zeros((frame_on.shape[0], frame_on.shape[1]), np.uint8)
    cv2.fillPoly(new_mask, [np.int32(destination)], (255, 255, 255))
    mask_inverse = cv2.bitwise_not(new_mask)
    augmented_image = cv2.bitwise_and(augmented_image, augmented_image, mask=mask_inverse)
    augmented_image = cv2.bitwise_or(warped_frame_from, augmented_image)
    return augmented_image


def play_video(captured_video):
    for frame_index in range(int(captured_video.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = captured_video.read()

        cv2.imshow("Video Capture", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    captured_video.release()
    cv2.destroyAllWindows()


def show_img(image):
    cv2.imshow("Image", image)
    cv2.waitKey()


def run(video_to_write_on, video_to_be_written, target_img, output_path, crop=False):
    if (video_to_be_written is None) or (video_to_write_on is None) or (target_img is None):
        return None

    height = int(video_to_write_on.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_to_write_on.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = float(video_to_write_on.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    # get the sizes of the target image
    target_height, target_width, channel_target = target_img.shape

    # init SIFT to get the descriptors
    sift = cv2.SIFT_create()

    # get the key points and descriptors of the target image
    target_key_points, target_descriptor = sift.detectAndCompute(target_img, None)
    # every keypoint has an attribute pt that is the coordinate of that point.

    # loop over all the longer video
    while True:
        # get the current frames of both videos
        ret_1, frame_on = video_to_write_on.read()
        ret_2, frame_from = video_to_be_written.read()

        if ret_1 and ret_2:
            # crop frame
            if crop:
                from_height = int(video_to_be_written.get(cv2.CAP_PROP_FRAME_HEIGHT))
                from_width = int(video_to_be_written.get(cv2.CAP_PROP_FRAME_WIDTH))
                w1 = int(from_width / 2 - from_height * (target_width / target_height) / 2)
                w2 = int(from_width / 2 + from_height * (target_width / target_height) / 2)
                frame_from = frame_from[0:from_height, w1:w2]

            # resize the frame that is to be written from to the same size as the targeted image.
            frame_from = cv2.resize(frame_from, (target_width, target_height))

            # get the key points and the descriptor of that frame on which we will put upon
            video_on_key_points, video_on_descriptor = sift.detectAndCompute(frame_on, None)

            # draw upon the frame the key points of itself
            # frame_on = cv2.drawKeypoints(frame_on, video_on_key_points, None)
            # cv2.imshow("video to be written on", frame_on)

            # start the brute force matcher
            acceptable_matches = get_matches(target_descriptor, video_on_descriptor)

            # draws the matches between the target image and the video to write upon
            # img_features = cv2.drawMatches(target_img, target_key_points, frame_on, video_on_key_points,
            # acceptable_matches, None, flags=2)
            # cv2.imshow("hi", img_features)

            # if number of acceptable matches are bigger than 50 then proceed
            if len(acceptable_matches) >= 50:
                # build a list of four key points (corners) of the target image
                # points --> key points of the target image
                points = np.float32(
                    [[0, 0], [0, target_height], [target_width, target_height], [target_width, 0]]).reshape(
                    -1, 1, 2)

                # get the matrix H, and the destination points
                matrix, destination = get_matrix_and_destination(acceptable_matches, target_key_points,
                                                                 video_on_key_points,
                                                                 points)

                # draw a polygon around the area in the frame that is to be written upon indicating the transformation
                # happening to the points p of the target image affected with H the matrix to become destination p`
                # p` = p*H
                # img2 = cv2.polylines(frame_on, [np.int32(destination)], True, (255, 0, 255), 3)
                # cv2.imshow("ha", img2)

                # get the warped output of the second frame that is to be written transformed using the matrix H
                warped_frame_from = cv2.warpPerspective(frame_from, matrix, (frame_on.shape[1], frame_on.shape[0]))
                # show the warped video
                # cv2.imshow("warped output", warped_frame_from)

                # merge the images

                augmented_image = merge_frames(frame_on, warped_frame_from, destination)

                cv2.imshow("final output", augmented_image)
                video_writer.write(augmented_image)
            # the condition to quit the show press q
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cap_1 = cv2.VideoCapture(os.path.join("assignment_2_materials", "book.mov"))
    cap_2 = cv2.VideoCapture(os.path.join("assignment_2_materials", "ar_source.mov"))
    img = cv2.imread(os.path.join("assignment_2_materials", "cv_cover.jpg"))
    # replace target image in cap 1 with cap 2
    run(video_to_write_on=cap_1, video_to_be_written=cap_2, target_img=img,
        output_path=os.path.join('output', 'output.avi'))
    cap_1.release()
    cap_2.release()
