#!/usr/bin/env python3
"""
Real-time YOLO detection with attention heatmaps visualization.
Shows where YOLO pays attention during object detection.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ultralytics import YOLO
import time


class YOLOHeatmapVisualizer:
    def __init__(self, model_path: str, source: int = 0):
        """
        Initialize the YOLO heatmap visualizer.

        Args:
            model_path: Path to YOLO model (ONNX or PT)
            source: Camera source (0 for default webcam)
        """
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Setup matplotlib figure with 3 subplots
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle('YOLO Detection with Attention Heatmap', fontsize=14)

        # Subplot titles
        self.axes[0].set_title('Original + Detections')
        self.axes[1].set_title('Attention Heatmap')
        self.axes[2].set_title('Overlay')

        # Remove axis ticks
        for ax in self.axes:
            ax.axis('off')

        # Initialize image displays
        self.im_original = None
        self.im_heatmap = None
        self.im_overlay = None

        # FPS tracking
        self.fps = 0
        self.prev_time = time.time()

        plt.tight_layout()

    def generate_attention_heatmap(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Generate attention heatmap based on detection results.

        Creates a heatmap showing where YOLO focuses attention:
        - Detection boxes get high attention
        - Confidence scores affect intensity
        - Gaussian blur creates smooth attention regions
        """
        h, w = frame.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # Get box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())

                # Clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Create attention region for this detection
                # Center of the detection
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                box_w, box_h = x2 - x1, y2 - y1

                # Create Gaussian attention centered on detection
                sigma_x = max(box_w // 2, 20)
                sigma_y = max(box_h // 2, 20)

                # Generate coordinate grids
                y_grid, x_grid = np.ogrid[:h, :w]

                # Gaussian distribution centered on detection
                gaussian = np.exp(-((x_grid - cx)**2 / (2 * sigma_x**2) +
                                   (y_grid - cy)**2 / (2 * sigma_y**2)))

                # Weight by confidence
                heatmap += gaussian * conf

                # Also add higher intensity inside the box
                heatmap[y1:y2, x1:x2] += conf * 0.5

        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Apply Gaussian blur for smoother visualization
        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)

        return heatmap

    def create_colored_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Convert grayscale heatmap to colored visualization."""
        # Normalize to 0-255
        heatmap_normalized = (heatmap * 255).astype(np.uint8)

        # Apply colormap (JET for typical heatmap look)
        colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # Convert BGR to RGB for matplotlib
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        return colored

    def create_overlay(self, frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Create overlay of original frame with heatmap."""
        colored_heatmap = self.create_colored_heatmap(heatmap)

        # Blend original frame with heatmap
        overlay = cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)

        return overlay

    def draw_detections(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        annotated = frame.copy()

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())

                # Get class name
                class_name = self.model.names[cls] if cls in self.model.names else f"class_{cls}"

                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated

    def update(self, frame_num=None):
        """Update function for animation."""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO detection
        results = self.model.predict(frame_rgb, verbose=False, conf=0.25)

        # Generate attention heatmap
        heatmap = self.generate_attention_heatmap(frame_rgb, results)

        # Create visualizations
        frame_with_detections = self.draw_detections(frame_rgb, results)
        colored_heatmap = self.create_colored_heatmap(heatmap)
        overlay = self.create_overlay(frame_rgb, heatmap, alpha=0.4)

        # Draw detections on overlay too
        overlay = self.draw_detections(overlay, results)

        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time

        # Add FPS to images
        cv2.putText(frame_with_detections, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Update matplotlib displays
        if self.im_original is None:
            self.im_original = self.axes[0].imshow(frame_with_detections)
            self.im_heatmap = self.axes[1].imshow(colored_heatmap)
            self.im_overlay = self.axes[2].imshow(overlay)
        else:
            self.im_original.set_data(frame_with_detections)
            self.im_heatmap.set_data(colored_heatmap)
            self.im_overlay.set_data(overlay)

        # Update detection count in title
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        self.fig.suptitle(f'YOLO Detection with Attention Heatmap | Detections: {num_detections} | FPS: {self.fps:.1f}',
                        fontsize=14)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        return self.im_original, self.im_heatmap, self.im_overlay

    def run(self):
        """Run the real-time visualization."""
        print("Starting YOLO Heatmap Visualization...")
        print("Press 'q' in the matplotlib window or close it to quit.")
        print("-" * 50)

        try:
            while plt.fignum_exists(self.fig.number):
                self.update()
                plt.pause(0.001)

                # Check for keyboard interrupt
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources."""
        self.cap.release()
        plt.close(self.fig)
        cv2.destroyAllWindows()
        print("Cleanup complete.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='YOLO Detection with Real-time Attention Heatmaps')
    parser.add_argument('--model', type=str,
                       default='/home/flybozon/object_detection/runs/detect/runs/detect/red_ball/weights/best.onnx',
                       help='Path to YOLO model (ONNX or PT)')
    parser.add_argument('--source', type=int, default=0,
                       help='Camera source (default: 0)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')

    args = parser.parse_args()

    print("=" * 60)
    print("YOLO Real-time Attention Heatmap Visualization")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Source: Camera {args.source}")
    print(f"Confidence threshold: {args.conf}")
    print("=" * 60)

    visualizer = YOLOHeatmapVisualizer(
        model_path=args.model,
        source=args.source
    )

    visualizer.run()


if __name__ == "__main__":
    main()
