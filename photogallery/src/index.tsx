import { furnituresPictures } from "../Pictures/furnitures/furnituresPictures.tsx";

import { root } from "@lynx-js/react";
import Gallery from "./Gallery.tsx";

function PictureList() {
  return <view className="gallery-wrapper multi-card">
  <Gallery pictureData={furnituresPictures} />;

  </view>
}

root.render(<PictureList />);

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept()
}
