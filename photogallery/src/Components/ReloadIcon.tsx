import "../../index.css";
import { useState } from "@lynx-js/react";
import reloadIcon from "../../Pictures/reload_icon.png";
import type { Picture } from "../../Pictures/faces/facesPictures.tsx";

type ReloadIconProps = {
  picture: Picture; // Add the Picture type as a prop
};

// Use this to clear the bounding boxes from the modal-content picture
export default function ReloadIcon({ picture }: ReloadIconProps) {
  const [isTapped, setIsTapped] = useState(false);
  const onTap = () => {
    if (picture.detected_bounding_boxes) {
      picture.detected_bounding_boxes.length = 0; // Clear detected bounding boxes
    }
    if (picture.added_bounding_boxes) {
      picture.added_bounding_boxes.length = 0;
    }
    if (picture.added_labels) {
      picture.added_labels.length = 0;
    }
    setIsTapped(false);
    setTimeout(() => setIsTapped(true), 0); // Wait till next tick to reset isTapped, to trigger re-render
  };
  return (
    <view className="reload-icon" bindtap={onTap}>
      {isTapped && <view className="circle" />}
      <image src={reloadIcon} className="icon" />
    </view>
  );
}
