import "../index.css";
import type { Picture } from "../Pictures/faces/facesPictures.tsx";
import blackImage from "../Pictures/Black_colour.jpg";

type ImageCardProps = {
  picture: Picture;
};

export default function ImageCard({ picture }: ImageCardProps) {
  return (
    <view className="picture-wrapper">
      <image
        className="image"
        style={{ borderRadius: "20px", width: "100%", aspectRatio: picture.width / picture.height, padding: "10px" }}
        src={picture.src}
      />

      {picture.detected_bounding_boxes?.map((coords: number[], index: number) => (
        <image
          className="image"
          style={{
            top: coords[1] * 100 + "%", left: coords[0] * 100 + "%", position: "absolute", width: (coords[2] - coords[0]) * 100 + "%", height: (coords[3] - coords[1]) * 100 + "%"
          }}
          src={blackImage}
        />
      ))}
      {picture.added_bounding_boxes?.map((coords: number[], index: number) => (
        <image
          className="image"
          style={{
            top: coords[1] * 100 + "%", left: coords[0] * 100 + "%", position: "absolute", width: (coords[2] - coords[0]) * 100 + "%", height: (coords[3] - coords[1]) * 100 + "%"
          }}
          src={blackImage}
        />
      ))}
    </view>
  );
}