import os
import json
import threading
from queue import Queue
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFilter
import torch
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights
from facenet_pytorch import InceptionResnetV1

# Modern color palette
COLORS = {
    'primary': '#6366f1',
    'primary_hover': '#4f46e5',
    'secondary': '#8b5cf6',
    'accent': '#06b6d4',
    'success': '#10b981',
    'warning': '#f59e0b',
    'error': '#ef4444',
    'neutral_900': '#111827',
    'neutral_800': '#1f2937',
    'neutral_700': '#374151',
    'neutral_600': '#4b5563',
    'neutral_500': '#6b7280',
    'neutral_400': '#9ca3af',
    'neutral_300': '#d1d5db',
    'neutral_200': '#e5e7eb',
    'neutral_100': '#f3f4f6',
    'white': '#ffffff',
    'glass': 'rgba(255, 255, 255, 0.1)',
    'female': '#ec4899',
    'male': '#3b82f6'
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Configuration (update paths as needed)
face_model_path = r"C:\Users\ADMIN\OneDrive\Documents\hackathon\best_facenet_model.pth"
gender_model_path = r"C:\Users\ADMIN\OneDrive\Documents\hackathon\best_gender_model.pth"
train_folder = r"C:\Users\ADMIN\OneDrive\Documents\hackathon\train"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model initialization (keeping your existing logic)
try:
    train_dataset = datasets.ImageFolder(train_folder)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    num_classes = len(idx_to_class)

    face_model = InceptionResnetV1(classify=True, num_classes=num_classes).to(device)
    face_model.load_state_dict(torch.load(face_model_path, map_location=device, weights_only=True))
    face_model.eval()

    gender_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    gender_model.fc = torch.nn.Linear(gender_model.fc.in_features, 2)
    gender_model.load_state_dict(torch.load(gender_model_path, map_location=device, weights_only=True))
    gender_model = gender_model.to(device)
    gender_model.eval()

    models_loaded = True
except Exception as e:
    print(f"Model loading error: {e}")
    models_loaded = False

gender_classes = ["Female", "Male"]

face_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

gender_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


def create_premium_circular_image(image, size=(120, 120)):
    """Create a circular image with premium styling"""
    image = image.resize(size, Image.Resampling.LANCZOS)

    # Create circular mask
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)

    # Apply subtle blur for depth
    mask = mask.filter(ImageFilter.GaussianBlur(0.5))

    # Create output with alpha
    output = Image.new('RGBA', size, (0, 0, 0, 0))
    output.paste(image, (0, 0))
    output.putalpha(mask)

    return output


def batch_predict(images, batch_size=8, progress_callback=None):
    """Enhanced batch prediction with progress tracking"""
    if not models_loaded:
        return []

    results = []
    total_batches = (len(images) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(images), batch_size):
        batch_images = images[batch_idx:batch_idx + batch_size]

        try:
            face_tensors = torch.stack([face_transform(img) for img, _ in batch_images]).to(device)
            gender_tensors = torch.stack([gender_transform(img) for img, _ in batch_images]).to(device)

            with torch.no_grad():
                face_outputs = face_model(face_tensors)
                face_preds = torch.argmax(face_outputs, dim=1).cpu().numpy()

                gender_outputs = gender_model(gender_tensors)
                gender_preds = torch.argmax(gender_outputs, dim=1).cpu().numpy()

                for j, (face_pred, gender_pred) in enumerate(zip(face_preds, gender_preds)):
                    img_path = batch_images[j][1]
                    confidence_face = torch.softmax(face_outputs[j], dim=0).max().item()
                    confidence_gender = torch.softmax(gender_outputs[j], dim=0).max().item()

                    results.append({
                        "path": img_path,
                        "identity": idx_to_class[face_pred],
                        "gender": gender_classes[gender_pred],
                        "confidence_face": confidence_face,
                        "confidence_gender": confidence_gender
                    })
        except Exception as e:
            print(f"Prediction error: {e}")
            continue

        if progress_callback:
            progress = (batch_idx // batch_size + 1) / total_batches
            progress_callback(progress)

    return results


class PremiumButton(ctk.CTkButton):
    """Custom premium button with enhanced styling"""

    def __init__(self, master, **kwargs):
        # Default premium styling
        defaults = {
            'corner_radius': 12,
            'border_width': 0,
            'font': ('SF Pro Display', 14, 'normal'),
            'hover': True,
        }
        defaults.update(kwargs)
        super().__init__(master, **defaults)


class StatsCard(ctk.CTkFrame):
    """Premium stats card component"""

    def __init__(self, master, title, value, color=COLORS['primary'], **kwargs):
        super().__init__(master, corner_radius=16, **kwargs)
        self.setup_card(title, value, color)

    def setup_card(self, title, value, color):
        self.grid_columnconfigure(0, weight=1)

        # Value
        value_label = ctk.CTkLabel(
            self,
            text=str(value),
            font=('SF Pro Display', 32, 'bold'),
            text_color=color
        )
        value_label.grid(row=0, column=0, pady=(20, 5), sticky='ew')

        # Title
        title_label = ctk.CTkLabel(
            self,
            text=title,
            font=('SF Pro Display', 14, 'normal'),
            text_color=COLORS['neutral_400']
        )
        title_label.grid(row=1, column=0, pady=(0, 20), sticky='ew')


class ResultCard(ctk.CTkFrame):
    """Premium result card with glassmorphism effect"""

    def __init__(self, master, result_data, **kwargs):
        defaults = {
            'corner_radius': 20,
            'border_width': 1,
            'border_color': COLORS['neutral_700'],
            'fg_color': COLORS['neutral_800']
        }
        defaults.update(kwargs)
        super().__init__(master, **defaults)

        self.result_data = result_data
        self.setup_card()
        self.setup_hover_effects()

    def setup_card(self):
        self.grid_columnconfigure(0, weight=1)

        # Image container
        img_frame = ctk.CTkFrame(self, fg_color='transparent', corner_radius=0)
        img_frame.grid(row=0, column=0, pady=(20, 15), sticky='ew')

        try:
            img = Image.open(self.result_data["path"]).convert('RGB')
            circular_img = create_premium_circular_image(img, (100, 100))
            photo = ctk.CTkImage(light_image=circular_img, dark_image=circular_img, size=(100, 100))

            img_label = ctk.CTkLabel(img_frame, image=photo, text="")
            img_label.image = photo
            img_label.pack()
        except Exception as e:
            # Fallback for missing images
            placeholder = ctk.CTkLabel(
                img_frame,
                text="📷",
                font=('SF Pro Display', 48),
                width=100,
                height=100
            )
            placeholder.pack()

        # Identity
        identity_label = ctk.CTkLabel(
            self,
            text=self.result_data['identity'],
            font=('SF Pro Display', 16, 'bold'),
            text_color=COLORS['white']
        )
        identity_label.grid(row=1, column=0, pady=(0, 8), sticky='ew')

        # Gender with color coding
        gender_color = COLORS['female'] if self.result_data["gender"] == "Female" else COLORS['male']
        gender_label = ctk.CTkLabel(
            self,
            text=f"● {self.result_data['gender']}",
            font=('SF Pro Display', 14, 'normal'),
            text_color=gender_color
        )
        gender_label.grid(row=2, column=0, pady=(0, 8), sticky='ew')

        # Confidence indicators
        if 'confidence_face' in self.result_data:
            confidence_text = f"ID: {self.result_data['confidence_face']:.1%} | Gender: {self.result_data['confidence_gender']:.1%}"
            confidence_label = ctk.CTkLabel(
                self,
                text=confidence_text,
                font=('SF Pro Display', 12, 'normal'),
                text_color=COLORS['neutral_400']
            )
            confidence_label.grid(row=3, column=0, pady=(0, 20), sticky='ew')

    def setup_hover_effects(self):
        """Add subtle hover effects"""

        def on_enter(event):
            self.configure(border_color=COLORS['primary'])

        def on_leave(event):
            self.configure(border_color=COLORS['neutral_700'])

        self.bind("<Enter>", on_enter)
        self.bind("<Leave>", on_leave)


class ProgressOverlay(ctk.CTkToplevel):
    """Premium progress overlay"""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Processing...")
        self.geometry("400x200")
        self.resizable(False, False)

        # Center on parent
        self.transient(parent)
        self.grab_set()

        self.setup_ui()

    def setup_ui(self):
        main_frame = ctk.CTkFrame(self, corner_radius=20)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="Processing Images",
            font=('SF Pro Display', 20, 'bold')
        )
        title_label.pack(pady=(30, 10))

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            main_frame,
            width=300,
            height=8,
            corner_radius=4,
            progress_color=COLORS['primary']
        )
        self.progress_bar.pack(pady=20)
        self.progress_bar.set(0)

        # Status label
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Initializing...",
            font=('SF Pro Display', 14),
            text_color=COLORS['neutral_400']
        )
        self.status_label.pack(pady=(0, 30))

    def update_progress(self, progress, status="Processing..."):
        self.progress_bar.set(progress)
        self.status_label.configure(text=status)
        self.update()


class PremiumApp(ctk.CTk):
    """Main application with premium UI"""

    def __init__(self):
        super().__init__()

        self.title("FaceRecognition Pro")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.results = []
        self.stats = {'total': 0, 'male': 0, 'female': 0}

        self.setup_sidebar()
        self.setup_main_area()
        self.setup_header()

    def setup_sidebar(self):
        """Create premium sidebar"""
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Logo/Title
        title_frame = ctk.CTkFrame(self.sidebar, fg_color='transparent')
        title_frame.pack(fill='x', padx=20, pady=(30, 20))

        title_label = ctk.CTkLabel(
            title_frame,
            text="FaceRecognition",
            font=('SF Pro Display', 24, 'bold'),
            text_color=COLORS['primary']
        )
        title_label.pack()

        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Pro",
            font=('SF Pro Display', 16, 'normal'),
            text_color=COLORS['neutral_400']
        )
        subtitle_label.pack()

        # Upload button
        self.upload_btn = PremiumButton(
            self.sidebar,
            text="📁 Select Folder",
            command=self.upload_folder,
            fg_color=COLORS['primary'],
            hover_color=COLORS['primary_hover'],
            height=50,
            font=('SF Pro Display', 16, 'bold')
        )
        self.upload_btn.pack(fill='x', padx=20, pady=(20, 10))

        # Stats section
        stats_frame = ctk.CTkFrame(self.sidebar, corner_radius=16)
        stats_frame.pack(fill='x', padx=20, pady=20)

        stats_title = ctk.CTkLabel(
            stats_frame,
            text="Statistics",
            font=('SF Pro Display', 18, 'bold')
        )
        stats_title.pack(pady=(20, 10))

        # Stats cards
        self.total_card = StatsCard(stats_frame, "Total Images", "0", COLORS['accent'])
        self.total_card.pack(fill='x', padx=15, pady=5)

        self.male_card = StatsCard(stats_frame, "Male", "0", COLORS['male'])
        self.male_card.pack(fill='x', padx=15, pady=5)

        self.female_card = StatsCard(stats_frame, "Female", "0", COLORS['female'])
        self.female_card.pack(fill='x', padx=15, pady=(5, 20))

        # Device info
        device_frame = ctk.CTkFrame(self.sidebar, corner_radius=12)
        device_frame.pack(fill='x', padx=20, pady=(0, 20))

        device_label = ctk.CTkLabel(
            device_frame,
            text=f"🔧 Device: {device.type.upper()}",
            font=('SF Pro Display', 12, 'normal'),
            text_color=COLORS['success'] if device.type == 'cuda' else COLORS['neutral_400']
        )
        device_label.pack(pady=15)

    def setup_header(self):
        """Create header area"""
        self.header = ctk.CTkFrame(self.main_container, height=80, corner_radius=16)
        self.header.pack(fill='x', padx=20, pady=(20, 10))
        self.header.pack_propagate(False)

        # Welcome message
        welcome_label = ctk.CTkLabel(
            self.header,
            text="Welcome to FaceRecognition Pro",
            font=('SF Pro Display', 20, 'bold')
        )
        welcome_label.pack(side='left', padx=30, pady=25)

        # Action buttons
        btn_frame = ctk.CTkFrame(self.header, fg_color='transparent')
        btn_frame.pack(side='right', padx=30, pady=20)

        export_btn = PremiumButton(
            btn_frame,
            text="📊 Export Results",
            command=self.export_results,
            fg_color=COLORS['secondary'],
            hover_color='#7c3aed',
            width=140,
            height=40
        )
        export_btn.pack(side='right', padx=(10, 0))

        clear_btn = PremiumButton(
            btn_frame,
            text="🗑️ Clear",
            command=self.clear_results,
            fg_color=COLORS['neutral_700'],
            hover_color=COLORS['neutral_600'],
            width=100,
            height=40
        )
        clear_btn.pack(side='right')

    def setup_main_area(self):
        """Create main content area"""
        self.main_container = ctk.CTkFrame(self, corner_radius=0)
        self.main_container.grid(row=0, column=1, sticky="nsew")

        # Scrollable results area
        self.results_frame = ctk.CTkScrollableFrame(
            self.main_container,
            corner_radius=16,
            scrollbar_button_color=COLORS['neutral_600'],
            scrollbar_button_hover_color=COLORS['neutral_500']
        )
        self.results_frame.pack(fill='both', expand=True, padx=20, pady=(10, 20))

        # Configure grid for results
        for i in range(6):  # 6 columns for better layout
            self.results_frame.grid_columnconfigure(i, weight=1)

        # Empty state
        self.show_empty_state()

    def show_empty_state(self):
        """Show empty state when no results"""
        self.empty_frame = ctk.CTkFrame(self.results_frame, fg_color='transparent')
        self.empty_frame.grid(row=0, column=0, columnspan=6, pady=100)

        empty_icon = ctk.CTkLabel(
            self.empty_frame,
            text="📸",
            font=('SF Pro Display', 64)
        )
        empty_icon.pack(pady=(0, 20))

        empty_label = ctk.CTkLabel(
            self.empty_frame,
            text="No images loaded yet",
            font=('SF Pro Display', 18, 'normal'),
            text_color=COLORS['neutral_400']
        )
        empty_label.pack(pady=(0, 10))

        empty_sublabel = ctk.CTkLabel(
            self.empty_frame,
            text="Select a folder to get started with face recognition",
            font=('SF Pro Display', 14, 'normal'),
            text_color=COLORS['neutral_500']
        )
        empty_sublabel.pack()

    def upload_folder(self):
        """Enhanced folder upload with progress tracking"""
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        if not folder_path:
            return

        if not models_loaded:
            messagebox.showerror("Error", "Models not loaded. Please check model paths.")
            return

        # Get image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(image_extensions)
        ]

        if not image_files:
            messagebox.showwarning("Warning", "No image files found in the selected folder.")
            return

        # Show progress overlay
        progress_window = ProgressOverlay(self)

        def process_images():
            processed_images = []

            # Load images
            for i, path in enumerate(image_files):
                try:
                    img = Image.open(path).convert('RGB')
                    processed_images.append((img, path))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue

                # Update progress
                progress = (i + 1) / len(image_files) * 0.3  # 30% for loading
                self.after(0, lambda p=progress: progress_window.update_progress(p,
                                                                                 f"Loading images... {i + 1}/{len(image_files)}"))

            if not processed_images:
                self.after(0, lambda: messagebox.showerror("Error", "No images could be loaded."))
                self.after(0, progress_window.destroy)
                return

            # Predict
            def progress_callback(pred_progress):
                total_progress = 0.3 + (pred_progress * 0.7)  # 70% for prediction
                self.after(0, lambda: progress_window.update_progress(total_progress,
                                                                      f"Processing... {int(pred_progress * 100)}%"))

            self.results = batch_predict(processed_images, progress_callback=progress_callback)

            # Update UI
            self.after(0, self.update_stats)
            self.after(0, self.display_results)
            self.after(0, progress_window.destroy)

        # Run in thread
        thread = threading.Thread(target=process_images)
        thread.daemon = True
        thread.start()

    def update_stats(self):
        """Update statistics"""
        self.stats['total'] = len(self.results)
        self.stats['male'] = sum(1 for r in self.results if r['gender'] == 'Male')
        self.stats['female'] = sum(1 for r in self.results if r['gender'] == 'Female')

        # Update cards
        self.total_card.setup_card("Total Images", self.stats['total'], COLORS['accent'])
        self.male_card.setup_card("Male", self.stats['male'], COLORS['male'])
        self.female_card.setup_card("Female", self.stats['female'], COLORS['female'])

    def display_results(self):
        """Display results with animation - FIXED"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        if not self.results:
            self.show_empty_state()
            return

        # Display cards with staggered animation
        for i, result in enumerate(self.results):
            result_row = i // 6
            result_col = i % 6

            # Staggered display - capture variables in closure
            delay = i * 30
            self.after(delay, self._create_delayed_card, result, result_row, result_col)

    def _create_delayed_card(self, result, row, col):
        """Helper method to create card with proper variable capture"""
        self.add_result_card(result, row, col)

    def add_result_card(self, result, row, col):
        """Add a result card to the grid"""
        card = ResultCard(
            self.results_frame,
            result,
            width=200,
            height=240
        )
        card.grid(row=row, column=col, padx=10, pady=10, sticky='ew')

    def export_results(self):
        """Export results to JSON"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Results"
        )

        if file_path:
            try:
                export_data = {
                    'statistics': self.stats,
                    'results': self.results,
                    'device': str(device),
                    'timestamp': str(threading.current_thread().ident)
                }

                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)

                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {e}")

    def clear_results(self):
        """Clear all results"""
        if self.results:
            if messagebox.askyesno("Confirm", "Clear all results?"):
                self.results = []
                self.update_stats()
                self.display_results()


def main():
    """Main entry point"""
    app = PremiumApp()
    app.mainloop()


if __name__ == "__main__":
    main()