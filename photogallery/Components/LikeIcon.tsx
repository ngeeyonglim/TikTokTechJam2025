import { useState } from "@lynx-js/react";
import redHeart from "../Pictures/redHeart.png";
import whiteHeart from "../Pictures/whiteHeart.png";
import "../index.css";
import type { Picture } from "../Pictures/furnitures/furnituresPictures.tsx";
const SERVER_URL = "http://192.168.0.245:5000";
const POST_BOUNDING_BOXES_ENDPOINT = "/update_data";

type LikeIconProps = {
  picture: Picture; // Add the Picture type as a prop
};

async function postBoundingBoxes(labels: number[], new_bbs: number[][], image: string): Promise<void> {
  const URI = SERVER_URL + POST_BOUNDING_BOXES_ENDPOINT;
  try {
    const res = await fetch(URI, {
      method: "POST",
      headers: { "Content-Type": "application/json"},
      body: JSON.stringify({
        labels: labels,
        new_bbs: new_bbs,
        image: image
      })
    });
    await res.json();
  } catch (err) {
  }
}
// Use this to clear the bounding boxes from the modal-content picture
export default function LikeIcon({picture}: LikeIconProps) {
  const [isLiked, setIsLiked] = useState(false);
  const onTap = () => {
    setIsLiked(!isLiked);
    if (picture.added_bounding_boxes && picture.added_labels) {
      postBoundingBoxes(picture.added_labels, picture.added_bounding_boxes, picture.localSrc);
    }
    if (picture.detected_bounding_boxes){
      picture.detected_bounding_boxes.length = 0; // Clear bounding boxes when liked/unliked
    }
    if (picture.added_bounding_boxes){
      picture.added_bounding_boxes.length = 0; // Clear bounding boxes when liked/unliked
    }
    if (picture.added_labels){
      picture.added_labels.length = 0; // Clear labels when liked/unliked
    }
  };
  return (
    <view className="like-icon" bindtap={onTap}>
      {isLiked && <view className="circle" />}
      {isLiked && <view className="circle circleAfter" />}
      <image src={isLiked ? redHeart : whiteHeart} className="heart-love" />
    </view>
  );
}
