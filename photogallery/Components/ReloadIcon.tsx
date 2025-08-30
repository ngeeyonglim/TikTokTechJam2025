import "../index.css";
import reloadIcon from "../Pictures/reload_icon.png";
import type { Picture } from "../Pictures/furnitures/furnituresPictures.tsx";

type ReloadIconProps = {
  picture: Picture; // Add the Picture type as a prop
};

// Use this to clear the bounding boxes from the modal-content picture
export default function ReloadIcon({picture}: ReloadIconProps) {
  const onTap = () => {
    if (picture.detected_bounding_boxes){
      picture.detected_bounding_boxes.length = 0; // Clear detected bounding boxes
    }
    if (picture.added_bounding_boxes){
      picture.added_bounding_boxes.length = 0;
    }
    if (picture.added_labels){
      picture.added_labels.length = 0;
    }
  };
  return (
    <view className="reload-icon" bindtap={onTap}>
      <image src={reloadIcon} className="icon" />
    </view>
  );
}
