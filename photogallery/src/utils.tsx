export const calculateEstimatedSize = (
  pictureWidth: number,
  pictureHeight: number,
) => {
  // Fixed styles of the gallery
  const galleryPadding = 20;
  const galleryMainAxisGap = 10;
  const gallerySpanCount = 2;
  const galleryWidth = 375;
  // Calculate the width of each ImageCard and return the relative height of the it.
  const itemWidth =
    (galleryWidth - galleryPadding * 2 - galleryMainAxisGap) / gallerySpanCount;
  return (itemWidth / pictureWidth) * pictureHeight;
};
