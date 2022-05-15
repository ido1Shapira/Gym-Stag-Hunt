class Shrub {
    constructor(tile, dimensions, position, value, valueToView, iconPath) {
        this.tile = tile;
        this.dimensions	= dimensions;
        this.position = position;
        this.value = value;
        this.valueToView = valueToView;
        this.iconImage = new Image();
        this.iconImage.src = ('data/assets/entities/'+iconPath);
    }
}