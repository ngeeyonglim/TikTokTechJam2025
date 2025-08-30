import "../index.css";
import uploadIcon from "../Pictures/upload_icon.png";
import type { Picture } from "../Pictures/faces/facesPictures.tsx";


const SERVER_URL = "http://192.168.0.245:5000";
const POST_BOUNDING_BOXES_ENDPOINT = "/update_data";

type UploadIconProps = {
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
// Use this to upload added bounding boxes to local AI model
export default function UploadIcon({picture}: UploadIconProps) {
  const onTap = () => {
    if (picture.added_bounding_boxes && picture.added_labels) {
      postBoundingBoxes(picture.added_labels, picture.added_bounding_boxes, picture.localSrc);
    }
  };
  return (
    <view className="upload-icon" bindtap={onTap}>
      <image src={uploadIcon} className="icon" />
    </view>
  );
}
