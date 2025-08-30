import "../index.css";
import { useState } from "@lynx-js/react";
import type { Picture } from "../Pictures/furnitures/furnituresPictures.tsx";

type ImageCardProps = {
  picture: Picture;
};

export default function ImageCard({picture}: ImageCardProps) {
  const reRender = useState(false)[1]; // We don't use the state variable, just the reRender function
  return (
    <view className="picture-wrapper"
    bindtap={() => {
      reRender((prev) => !prev); // Trigger a re-render to show/hide the bounding boxes
    }}
    >
      <image
        className="image"
        style={{ borderRadius: "20px", width: "100%", aspectRatio: picture.width / picture.height, padding: "10px" }}
        src={picture.src}
      />

      { picture.detected_bounding_boxes?.map((coords: number[], index: number) => (
          <image
            blur-radius="20" // apply default blur to whole image
            className="image"
            style={{
              top: coords[1]*100 + "%", left: coords[0]*100 + "%", position: "absolute", width: (coords[2] - coords[0])*100 + "%", height: (coords[3]-coords[1])*100 + "%"}}
            src={picture.src} // TODO: replace with black image
          />
        )) }
      { picture.added_bounding_boxes?.map((coords: number[], index: number) => (
          <image
            blur-radius="20" // apply default blur to whole image
            className="image"
            style={{
              top: coords[1]*100 + "%", left: coords[0]*100 + "%", position: "absolute", width: (coords[2] - coords[0])*100 + "%", height: (coords[3]-coords[1])*100 + "%"}}
            src={picture.src} // TODO: replace with black image
          />
        )) }
    </view>
  );
}