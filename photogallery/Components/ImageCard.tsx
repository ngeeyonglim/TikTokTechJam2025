import "../index.css";
import type { Picture } from "../Pictures/furnitures/furnituresPictures.tsx";
import LikeIcon from "./LikeIcon.tsx";

type ImageCardProps = {
  picture: Picture;
};

export default function ImageCard({picture}: ImageCardProps) {
  return (
    <view className="picture-wrapper"
    >
      
      <image
        className="image"
        style={{ width: "100%", aspectRatio: picture.width / picture.height }}
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

      <LikeIcon picture={picture}/>
    </view>
  );
}
