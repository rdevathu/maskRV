import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import pandas as pd
from scipy import interpolate
import sys
import os
import json


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class RVAnnotationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("DrawRV")
        
        # Set app icon
        icon = tk.PhotoImage(file=resource_path("rv_icon.png"))
        self.master.iconphoto(False, icon)

        # Initialize variables
        self.frames = []
        self.current_frame_index = 0
        self.points = []
        self.nodes = []
        self.annotations = {}
        self.rejected_frames = set()
        self.df = None
        self.video_filename = ""
        self.avi_quality = tk.StringVar(value="Normal")  # Default quality
        self.progress_var = tk.DoubleVar()

        self.point_colors = {
            "RV_Apex": "green",
            "Lateral_Annulus": "purple",
            "Septal_Annulus": "yellow"
        }
        self.anatomical_points = ["RV_Apex", "Lateral_Annulus", "Septal_Annulus"]

        # Create output directories in a writable location
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.frames_dir = None
        self.masks_dir = None
        self.output_dir = None

        self._create_widgets()
        self._setup_canvas_layers()

        self.master.bind('<Left>', self.previous_frame)
        self.master.bind('<Right>', self.next_frame)
        self.master.bind('1', lambda event: self.set_avi_quality("Poor"))
        self.master.bind('2', lambda event: self.set_avi_quality("Adequate"))
        self.master.bind('3', lambda event: self.set_avi_quality("Excellent"))
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Handle the window close event."""
        if messagebox.askokcancel("Quit", "Do you want to save your work and quit?"):
            self.save_all_and_quit()
        else:
            self.master.destroy()

    def _create_widgets(self):
        """Create and setup all GUI widgets."""
        # Create a frame for the progress bar and frame counter at the top
        self.progress_frame = tk.Frame(self.master)
        self.progress_frame.pack(fill=tk.X, padx=10, pady=5)

        # Create and pack the progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100, style="TProgressbar")
        self.progress_bar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(10, 0))

        self.style = ttk.Style()
        self.style.configure("TProgressbar", foreground='blue', background='blue')

        # Create and pack the frame counter label
        self.frame_counter = tk.Label(self.progress_frame, text="0/0", width=5)
        self.frame_counter.pack(side=tk.RIGHT, padx=(0, 0))

        # Create canvas for image display
        self.canvas = tk.Canvas(self.master, width=800, height=600)
        self.canvas.pack()

        # Instruction label
        self.instruction_label = tk.Label(self.master, text="Load an AVI file to begin", font=("Arial", 14))
        self.instruction_label.pack()

        # AVI quality radio buttons
        quality_frame = tk.Frame(self.master)
        quality_frame.pack(pady=10)
        
        tk.Label(quality_frame, text="AVI Quality:").pack(side=tk.LEFT)
        
        qualities = [("Poor", "Poor"), ("Adequate", "Adequate"), ("Excellent", "Excellent")]
        for text, value in qualities:
            tk.Radiobutton(quality_frame, text=text, variable=self.avi_quality, value=value).pack(side=tk.LEFT)

        # Button frame for horizontal alignment
        button_frame = tk.Frame(self.master)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        # Load video button
        self.btn_load = tk.Button(button_frame, text="Load Video", command=self.load_video)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        # Back button
        self.btn_back = tk.Button(button_frame, text="Back", command=self.previous_frame, state=tk.DISABLED)
        self.btn_back.pack(side=tk.LEFT, padx=5)

        # Forward button
        self.btn_forward = tk.Button(button_frame, text="Forward", command=self.next_frame, state=tk.DISABLED)
        self.btn_forward.pack(side=tk.LEFT, padx=5)

        # Reject frame button
        self.btn_reject = tk.Button(button_frame, text="Reject Frame", command=self.reject_frame, fg="#FF9999")
        self.btn_reject.pack(side=tk.LEFT, padx=5)

        # Save All and Quit button
        self.btn_save_all = tk.Button(button_frame, text="Save All and Quit", command=self.save_all_and_quit, fg="#99FF99")
        self.btn_save_all.pack(side=tk.LEFT, padx=5)

    def update_avi_quality(self, event=None):
        value = self.quality_slider.get()
        if value < 0.67:
            self.avi_quality = "Poor"
        elif value < 1.33:
            self.avi_quality = "Adequate"
        else:
            self.avi_quality = "Excellent"
        self.quality_label.config(text=self.avi_quality)

    def _setup_canvas_layers(self):
        """Setup canvas layers for proper stacking of elements."""
        self.canvas.create_rectangle(0, 0, 1, 1, tags="image_layer")
        self.canvas.create_rectangle(0, 0, 1, 1, tags="spline_layer")
        self.canvas.create_rectangle(0, 0, 1, 1, tags="node_layer")

        self.canvas.tag_raise("spline_layer", "image_layer")
        self.canvas.tag_raise("node_layer", "spline_layer")

    def load_video(self):
        """Load an AVI video file and prepare frames for annotation."""
        # Check if there's an existing video loaded
        if self.video_filename:
            # Save current work
            self.save_all_and_quit(quit_app=False)
            
            # Reset necessary variables
            self.frames = []
            self.current_frame_index = 0
            self.points = []
            self.nodes = []
            self.annotations = {}
            self.rejected_frames = set()
            self.df = None
            self.video_filename = ""
            
        file_path = filedialog.askopenfilename(filetypes=[("AVI files", "*.avi")])
        if not file_path:
            return

        self.video_filename = os.path.splitext(os.path.basename(file_path))[0]

        # Create output directories in the same folder as the script
        self.output_dir = os.path.join(self.script_dir, self.video_filename)
        self.frames_dir = os.path.join(self.output_dir, "frames")
        self.masks_dir = os.path.join(self.output_dir, "masks")
        
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)

        cap = cv2.VideoCapture(file_path)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()

        total_frames = len(all_frames)
        if total_frames < 64:
            # If less than 64 frames, take all frames up to 32
            self.frames = all_frames[:min(32, total_frames)]
        else:
            # If 64 or more frames, take every other frame from the first 64
            self.frames = all_frames[0:64:2]

        self.current_frame_index = 0
        self.show_frame()
        self.instruction_label.config(text="Choose RV Apex")
        self.btn_back.config(state=tk.NORMAL)
        self.btn_forward.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.update_frame_counter()
        self.update_progress()

    def show_frame(self):
        """Display the current frame on the canvas and restore annotations if available."""
        if self.current_frame_index >= len(self.frames):
            self.finish_annotation()
            return

        frame = self.frames[self.current_frame_index]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        new_w = 800
        new_h = int(new_w / aspect_ratio)
        if new_h > 600:
            new_h = 600
            new_w = int(new_h * aspect_ratio)
        
        display_frame = cv2.resize(frame, (new_w, new_h))
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(display_frame))
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all")
        self._setup_canvas_layers()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo, tags=("image", "image_layer"))
        self.canvas.bind("<Button-1>", self.add_point)

        # Restore annotations if available
        if self.current_frame_index in self.annotations:
            self.restore_annotation(self.annotations[self.current_frame_index])
        else:
            self.points = []
            self.nodes = []
            self.instruction_label.config(text="Choose RV Apex")

        self.update_all()

    def restore_annotation(self, annotation):
        """Restore a previously saved annotation."""
        self.points = annotation['points']
        self.nodes = []
        
        # Create a mapping of coordinates to types
        coord_to_type = {(p["coords"][0], p["coords"][1]): p["type"] for p in self.points}
        
        for i, point in enumerate(annotation['spline_points']):
            x, y = point
            if (x, y) in coord_to_type:
                color = self.point_colors[coord_to_type[(x, y)]]
            else:
                color = "blue"
            node = self.canvas.create_oval(x-10, y-10, x+10, y+10, fill=color, tags=("node", "node_layer"))
            self.nodes.append(node)
        
        self.draw_shape()
        self.canvas.tag_bind("node", "<B1-Motion>", self.move_node)
        self.instruction_label.config(text="Edit annotation as needed")

    def add_point(self, event):
        """Add a point to the annotation when the user clicks on the image."""
        if len(self.points) < 3:
            x, y = event.x, event.y
            anatomical_point = self.anatomical_points[len(self.points)]
            color = self.point_colors[anatomical_point]
            node = self.canvas.create_oval(x-10, y-10, x+10, y+10, fill=color, tags=("node", "node_layer"))
            self.points.append({"coords": (x, y), "type": anatomical_point, "color": color})
            self.nodes.append(node)
            
            if len(self.points) == 1:
                self.instruction_label.config(text="Choose Lateral Annulus")
            elif len(self.points) == 2:
                self.instruction_label.config(text="Choose Septal Annulus")
            elif len(self.points) == 3:
                self.create_initial_shape()
                self.instruction_label.config(text="Drag nodes to adjust the RV contour")

    def create_initial_shape(self):
        """Create the initial shape with additional control points."""
        if len(self.points) < 3:
            return  # Not enough points to create a shape yet

        new_nodes = []
        
        # Add 4 nodes between lateral annulus and apex
        p1 = next(p["coords"] for p in self.points if p["type"] == "Lateral_Annulus")
        p2 = next(p["coords"] for p in self.points if p["type"] == "RV_Apex")
        for i in range(1, 5):
            t = i / 5
            x = int((1-t)*p1[0] + t*p2[0])
            y = int((1-t)*p1[1] + t*p2[1])
            node = self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="blue", tags=("node", "node_layer"))
            new_nodes.append(node)
        
        # Add 3 nodes between septal annulus and apex
        p1 = next(p["coords"] for p in self.points if p["type"] == "Septal_Annulus")
        for i in range(3, 0, -1):  # Reverse order to go from septal annulus to apex
            t = i / 4
            x = int((1-t)*p1[0] + t*p2[0])
            y = int((1-t)*p1[1] + t*p2[1])
            node = self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="blue", tags=("node", "node_layer"))
            new_nodes.append(node)

        # Add 1 node between the annulus points
        p1 = next(p["coords"] for p in self.points if p["type"] == "Lateral_Annulus")
        p2 = next(p["coords"] for p in self.points if p["type"] == "Septal_Annulus")
        x = int((p1[0] + p2[0]) / 2)
        y = int((p1[1] + p2[1]) / 2)
        node = self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="blue", tags=("node", "node_layer"))
        new_nodes.append(node)

        # Insert new nodes into self.nodes
        lateral_annulus_index = next(i for i, p in enumerate(self.points) if p["type"] == "Lateral_Annulus")
        rv_apex_index = next(i for i, p in enumerate(self.points) if p["type"] == "RV_Apex")
        septal_annulus_index = next(i for i, p in enumerate(self.points) if p["type"] == "Septal_Annulus")
        
        self.nodes = [self.nodes[lateral_annulus_index]] + new_nodes[:4] + [self.nodes[rv_apex_index]] + new_nodes[4:7] + [self.nodes[septal_annulus_index]] + [new_nodes[7]]

        self.draw_shape()
        self.canvas.tag_bind("node", "<B1-Motion>", self.move_node)

    def move_node(self, event):
        """Handle node movement when the user drags a node."""
        item = self.canvas.find_withtag("current")[0]
        if item in self.nodes:
            self.canvas.coords(item, event.x-10, event.y-10, event.x+10, event.y+10)
            
            # Update the corresponding point in self.points if it's an anatomical point
            for point in self.points:
                if self.canvas.itemcget(item, "fill") == self.point_colors[point["type"]]:
                    point["coords"] = (event.x, event.y)
                    break
            
            self.draw_shape()

    def draw_shape(self):
        """Draw the shape based on the current node positions."""
        self.canvas.delete("spline")
        if len(self.nodes) < 3:
            return  # Not enough points to draw anything yet

        points = [self.canvas.coords(node)[:2] for node in self.nodes]
        points = [(x+10, y+10) for x, y in points]  # Adjust for node size

        # Create a single continuous spline
        if len(points) >= 4:  # We need at least 4 points for a meaningful spline
            try:
                # Prepare points for a closed spline
                x, y = zip(*points)
                x = list(x) + [x[0]]  # Add the first point to the end to close the curve
                y = list(y) + [y[0]]

                # Create smooth spline curve
                tck, u = interpolate.splprep([x, y], s=0, per=True)  # per=True for a periodic (closed) spline
                unew = np.linspace(0, 1, 200)
                smooth_points = interpolate.splev(unew, tck)

                # Draw the smooth curve
                self.canvas.create_line(list(zip(smooth_points[0], smooth_points[1])), 
                                        fill="red", tags=("spline", "spline_layer"), smooth=True, width=2)
            except ValueError:
                # If spline creation fails, fall back to drawing lines
                for i in range(len(points)):
                    self.canvas.create_line(points[i], points[(i+1) % len(points)], 
                                            fill="red", tags=("spline", "spline_layer"), width=2)
        else:
            # If we don't have enough points for a spline, draw simple lines
            for i in range(len(points)):
                self.canvas.create_line(points[i], points[(i+1) % len(points)], 
                                        fill="red", tags=("spline", "spline_layer"), width=2)

        # Draw a straight line between annulus points
        if len(self.points) >= 2:
            self.canvas.create_line(points[0], points[-2], 
                                    fill="magenta", tags=("spline", "spline_layer"), width=2)


    def save_current_frame(self):
        """Save the current frame's annotation."""
        if len(self.nodes) < 3:
            return  # Not enough points to save

        orig_h, orig_w = self.frames[self.current_frame_index].shape[:2]
        display_w, display_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        
        scale_x, scale_y = orig_w / display_w, orig_h / display_h
        
        # Get all node coordinates
        all_points = [self.canvas.coords(node)[:2] for node in self.nodes]
        all_points = [(x+10, y+10) for x, y in all_points]  # Adjust for node size
        
        # Update self.points with the current positions
        for point in self.points:
            for node in self.nodes:
                if self.canvas.itemcget(node, "fill") == self.point_colors[point["type"]]:
                    x, y = self.canvas.coords(node)[:2]
                    point["coords"] = (x+10, y+10)  # Adjust for node size
                    break
        
        # Create anatomical_points dictionary with scaled coordinates
        anatomical_points = {
            point["type"]: (int(point["coords"][0] * scale_x), int(point["coords"][1] * scale_y))
            for point in self.points
        }
        
        # Create a smooth spline curve
        x, y = zip(*all_points)
        x = list(x) + [x[0]]  # Add the first point to the end to close the curve
        y = list(y) + [y[0]]
        
        tck, u = interpolate.splprep([x, y], s=0, per=True)
        unew = np.linspace(0, 1, 200)
        smooth_points = interpolate.splev(unew, tck)
        
        # Scale the smooth points
        scaled_smooth_points = [(int(x*scale_x), int(y*scale_y)) for x, y in zip(smooth_points[0], smooth_points[1])]
        
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        pts = np.array(scaled_smooth_points, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(mask, [pts], 255)

        frame_filename = f"{self.video_filename}_frame_{self.current_frame_index}.png"
        mask_filename = f"{self.video_filename}_mask_{self.current_frame_index}.png"

        cv2.imwrite(os.path.join(self.frames_dir, frame_filename), self.frames[self.current_frame_index])
        cv2.imwrite(os.path.join(self.masks_dir, mask_filename), mask)

        # Save the spline representation
        spline_points = [(int(x*scale_x), int(y*scale_y)) for x, y in all_points]
        spline_points_json = json.dumps(spline_points)

        # Save annotation for the current frame
        self.annotations[self.current_frame_index] = {
            'Frame': frame_filename,
            'Mask': mask_filename,
            'RV_Apex': anatomical_points["RV_Apex"],
            'Lateral_Annulus': anatomical_points["Lateral_Annulus"],
            'Septal_Annulus': anatomical_points["Septal_Annulus"],
            'Spline_Points': spline_points_json,
            'points': self.points,
            'spline_points': all_points
        }

        # Remove from rejected frames if it was previously rejected
        if self.current_frame_index in self.rejected_frames:
            self.rejected_frames.remove(self.current_frame_index)


    def previous_frame(self, event=None):
        """Move to the previous frame and save the current frame."""
        self.save_current_frame()
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.show_frame()
        else:
            pass
        self.update_all()

    def next_frame(self, event=None):
        """Move to the next frame and save the current frame."""
        self.save_current_frame()
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.show_frame()
        else:
            pass
        self.update_all()

    def reject_frame(self):
        """Reject the current frame and move to the next one."""
        self.rejected_frames.add(self.current_frame_index)
        
        # Remove annotation for the rejected frame if it exists
        if self.current_frame_index in self.annotations:
            del self.annotations[self.current_frame_index]
        
        # Clear the canvas
        self.canvas.delete("all")
        self._setup_canvas_layers()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo, tags=("image", "image_layer"))
        
        self.points = []
        self.nodes = []
        self.instruction_label.config(text="Frame rejected")
        
        self.next_frame()

    def save_all_and_quit(self, quit_app=True):
        """Save all annotations and quit the application."""
        # First, check if there's anything to save
        if not self.video_filename:
            if quit_app:
                self.master.quit()
            return

        self.save_current_frame()  # Save the current frame before quitting
        
        # Create DataFrame from annotations and rejected frames
        data = []
        for i in range(len(self.frames)):
            frame_filename = f"{self.video_filename}_frame_{i}.png"
            if i in self.rejected_frames:
                data.append({
                    'Frame': frame_filename,
                    'Mask': None,
                    'Rejected': True,
                    'RV_Apex': None,
                    'Lateral_Annulus': None,
                    'Septal_Annulus': None,
                    'Spline_Points': None
                })
            elif i in self.annotations:
                anno = self.annotations[i]
                data.append({
                    'Frame': anno['Frame'],
                    'Mask': anno['Mask'],
                    'Rejected': False,
                    'RV_Apex': anno['RV_Apex'],
                    'Lateral_Annulus': anno['Lateral_Annulus'],
                    'Septal_Annulus': anno['Septal_Annulus'],
                    'Spline_Points': anno['Spline_Points']
                })
            else:
                # Handle frames that were neither annotated nor explicitly rejected
                data.append({
                    'Frame': frame_filename,
                    'Mask': None,
                    'Rejected': False,
                    'RV_Apex': None,
                    'Lateral_Annulus': None,
                    'Septal_Annulus': None,
                    'Spline_Points': None
                })

        self.df = pd.DataFrame(data)
        
        # Add AVI quality to the DataFrame
        self.df['AVI_Quality'] = self.avi_quality.get()
        
        # Save the DataFrame to CSV
        csv_path = os.path.join(self.output_dir, f"{self.video_filename}_manifest.csv")
        self.df.to_csv(csv_path, index=False)
        
        if quit_app:
            messagebox.showinfo("Finished", f"RV Annotation completed. Results saved to {csv_path}")
            self.master.quit()
        else:
            print(f"Results saved to {csv_path}")

    def update_all(self):
        self.update_navigation_buttons()
        self.update_progress()
        self.update_frame_counter()

    def update_navigation_buttons(self):
        """Update the state of navigation buttons based on current frame index."""
        self.btn_back.config(state=tk.NORMAL if self.current_frame_index > 0 else tk.DISABLED)
        self.btn_forward.config(state=tk.NORMAL if self.current_frame_index < len(self.frames) - 1 else tk.DISABLED)

    def update_progress(self):
        if self.frames:
            progress = (self.current_frame_index + 1) / len(self.frames) * 100
            self.progress_var.set(progress)

    def update_frame_counter(self):
        if self.frames:
            current = self.current_frame_index 
            total = len(self.frames) - 1
            self.frame_counter.config(text=f"{current}/{total}")

    def finish_annotation(self):
        """Finish the annotation process and save the results."""
        csv_path = os.path.join(self.output_dir, f"{self.video_filename}_manifest.csv")
        self.df.to_csv(csv_path, index=False)
        messagebox.showinfo("Finished", f"RV Annotation completed. Results saved to {csv_path}")
        self.master.quit()

    def recreate_spline(self, spline_points_json):
        """Recreate the spline from saved data."""
        spline_points = json.loads(spline_points_json)
        x, y = zip(*spline_points)
        x = list(x) + [x[0]]  # Add the first point to the end to close the curve
        y = list(y) + [y[0]]
        
        tck, u = interpolate.splprep([x, y], s=0, per=True)
        unew = np.linspace(0, 1, 200)
        smooth_points = interpolate.splev(unew, tck)
        
        return list(zip(smooth_points[0], smooth_points[1]))

    def set_avi_quality(self, quality):
        """Set the AVI quality."""
        self.avi_quality.set(quality)
    

if __name__ == "__main__":
    root = tk.Tk()
    app = RVAnnotationApp(root)
    root.lift()  # Raise the window
    root.focus_force()  # Force focus on the window
    root.mainloop()