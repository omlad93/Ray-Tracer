# Ray-Tracer
    Creating a png file from txt file describing a scene
    All Data-Structures are implemented in structure.py
    

# Class Scene
    Hold necessey data for performing ray tracing in fields as described below:
        self.general :  a Scene_Set class instance
        self.camera  :  a Camera class instance
        self.shapes  :  a dictnoray mapping strings to list of shapes by type
                        keys: 'boxes', 'planes', 'spheres'
                        for each shape type there is a Class 
        self.lights  :  a list of Light Class instances
        self.screen  :  a Screen Class instance

# Class Screen
    Hold necessey screen data for performing hit calculation:
        self.Z              :   Z Axis
        self.X              :   X Axis
        self.Y              :   Y Axis
        self.X_pixels       :   number of horizontal pixels
        self.Y_pixels       :   number of verical pixels 
        self.pixel_size     :   length of pixel's edges
        self.width          :   width of screen
        self.hight          :   hight of screen
        self.pixel_centers  :   a list of lists of np.arrays represing pixels center points
        self.pixel_rays     :   a list of lists of np.arrays represing rays through pixels
                                indexing is matching self.pixel_centers 
        