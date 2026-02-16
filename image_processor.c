#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include <ctype.h> // for isspace ve ungetc use

// Constants for Canny
#define PI 3.14159265
#define LOW_THRESHOLD_RATIO 0.09
#define HIGH_THRESHOLD_RATIO 0.18

// Structure and Prototypes 

// Define the structure to hold image data
typedef struct {
    int width;
    int height;
    int max_val; 
    unsigned char** pixels; 
} PGMImage;

// Function Prototypes
void display_menu();
int load_pgm_image(PGMImage* img, const char* filename);
int save_pgm_image(const PGMImage* img, const char* filename);
void free_image_memory(PGMImage* img);
int is_image_loaded(const PGMImage* img);

// Core operation wrappers
void resize_image(PGMImage* img);
void apply_filter(PGMImage* img);
void edge_detection(PGMImage* img);
void compute_lbp(PGMImage* img);

// Helper Prototypes 'const' parameter is here for warning removal)
void create_new_image(const PGMImage* original, PGMImage* new_img, int new_w, int new_h);
void deep_copy_image(const PGMImage* original, PGMImage* copy);
void sort_nine(unsigned char arr[9]);

// 2. Resizing
void nearest_neighbor_zoom(const PGMImage* original, PGMImage* new_img, int factor);
void subsample_shrink(const PGMImage* original, PGMImage* new_img, int factor);

// 3. Filtering
void average_filter(const PGMImage* original, PGMImage* new_img);
void median_filter(const PGMImage* original, PGMImage* new_img);

// 4. Edge Detection
void sobel_edge_detection(const PGMImage* original, PGMImage* new_img);
void prewitt_edge_detection(const PGMImage* original, PGMImage* new_img);
void canny_edge_detector(PGMImage* img);

// Canny helper prototypes
void gaussian_blur(const PGMImage* original, float** blurred);
void compute_gradient_and_magnitude(float** blurred, int W, int H, float** magnitude, float** angle);
void non_maximum_suppression(float** mag, float** angle, int W, int H, unsigned char** suppressed);
void hysteresis_thresholding(unsigned char** suppressed, int W, int H, unsigned char** final_edges);
void edge_tracking(unsigned char** edges, int x, int y, int W, int H, int high_thresh); 

// 5. LBP
void calculate_lbp(const PGMImage* original, PGMImage* new_img);

// helper function
// it is used to skip pgm folder headers spaces and comment lines
void skip_comments(FILE* f) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') { // skip comment line
            while ((c = fgetc(f)) != EOF && c != '\n' && c != '\r');
        } else if (!isspace(c)) { // if not space then add the char and stop
            ungetc(c, f);
            break;
        }
    }
}

// Main Function and Menu

int main() {
    PGMImage current_image = {0, 0, 0, NULL}; 
    int choice;
    char filename[256];

    printf("--- Welcome to the PGM Image Processor ---\n");

    do {
        display_menu();
        printf("Enter your choice: ");
        if (scanf("%d", &choice) != 1) {
            printf("Invalid input. Please enter a number.\n");
            while (getchar() != '\n');
            choice = -1;
            continue;
        }

        switch (choice) {
            case 1:
                printf("Enter input PGM file path: ");
                scanf("%s", filename);
                free_image_memory(&current_image); 
                load_pgm_image(&current_image, filename);
                break;
            case 2:
                if (is_image_loaded(&current_image)) resize_image(&current_image);
                break;
            case 3:
                if (is_image_loaded(&current_image)) apply_filter(&current_image);
                break;
            case 4:
                if (is_image_loaded(&current_image)) edge_detection(&current_image);
                break;
            case 5:
                if (is_image_loaded(&current_image)) compute_lbp(&current_image);
                break;
            case 6:
                if (is_image_loaded(&current_image)) {
                    printf("Enter output PGM file path: ");
                    scanf("%s", filename);
                    save_pgm_image(&current_image, filename);
                }
                break;
            case 0:
                printf("Exiting program. Goodbye!\n");
                break;
            default:
                printf("Invalid choice. Please select an option from 0 to 6.\n");
        }
    } while (choice != 0);

    free_image_memory(&current_image);
    return 0;
}

void display_menu() {
    printf("\n--- Operation Menu ---\n");
    printf("1 - Load PGM Image\n");
    printf("2 - Zoom/Shrink Image\n");
    printf("3 - Apply Filter (Average/Median)\n");
    printf("4 - Edge Detection (Sobel/Prewitt/Canny)\n");
    printf("5 - Compute Local Binary Pattern (LBP)\n");
    printf("6 - Save Processed Image\n");
    printf("0 - Exit\n");
    printf("----------------------\n");
}

// General Helpers

int is_image_loaded(const PGMImage* img) {
    if (img == NULL || img->pixels == NULL) {
        printf("ERROR: No image loaded. Please load an image first (Option 1).\n");
        return 0;
    }
    return 1;
}

void free_image_memory(PGMImage* img) {
    if (img->pixels != NULL) {
        for (int i = 0; i < img->height; i++) {
            free(img->pixels[i]);
        }
        free(img->pixels);
        img->pixels = NULL;
        img->width = 0;
        img->height = 0;
        img->max_val = 0;
    }
}

void create_new_image(const PGMImage* original, PGMImage* new_img, int new_w, int new_h) {
    new_img->width = new_w;
    new_img->height = new_h;
    new_img->max_val = original->max_val;

    new_img->pixels = (unsigned char**)malloc(new_h * sizeof(unsigned char*));
    if (new_img->pixels == NULL) return;
    for (int i = 0; i < new_h; i++) {
        new_img->pixels[i] = (unsigned char*)calloc(new_w, sizeof(unsigned char)); 
          if (new_img->pixels[i] == NULL) return;
    }
}

void deep_copy_image(const PGMImage* original, PGMImage* copy) {
    free_image_memory(copy); 
    create_new_image(original, copy, original->width, original->height);
    if (copy->pixels == NULL) return;
    for (int i = 0; i < original->height; i++) {
        memcpy(copy->pixels[i], original->pixels[i], original->width * sizeof(unsigned char));
    }
}

float** create_float_array(int H, int W) {
    float** arr = (float**)malloc(H * sizeof(float*));
    if (arr == NULL) return NULL;
    for (int i = 0; i < H; i++) {
        arr[i] = (float*)calloc(W, sizeof(float));
        if (arr[i] == NULL) {
             for (int j = 0; j < i; j++) free(arr[j]);
             free(arr);
             return NULL;
        }
    }
    return arr;
}

void free_float_array(float** arr, int H) {
    if (arr) {
        for (int i = 0; i < H; i++) {
            free(arr[i]);
        }
        free(arr);
    }
}

// 1. Load PGM File 

int load_pgm_image(PGMImage* img, const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        perror("Error opening file"); 
        return 0;
    }
    
    char magic[3];
    
    // (P5/P2) arrow
    if (fscanf(fp, "%2s", magic) != 1 || (strcmp(magic, "P5") != 0 && strcmp(magic, "P2") != 0)) {
        printf("ERROR: File is not a P5 (binary) or P2 (ascii) PGM format.\n"); 
        fclose(fp); 
        return 0;
    }
    
    skip_comments(fp);

    // read the dimensions
    if (fscanf(fp, "%d %d", &img->width, &img->height) != 2) {
        printf("ERROR: Invalid PGM dimensions.\n"); 
        fclose(fp); 
        return 0;
    }
    
    skip_comments(fp);

    // read max val
    if (fscanf(fp, "%d", &img->max_val) != 1) {
        printf("ERROR: Invalid PGM max_val.\n"); 
        fclose(fp); 
        return 0;
    }
    
    // skip the last line char
    fgetc(fp); 
    
    // memory allocation
    img->pixels = (unsigned char**)malloc(img->height * sizeof(unsigned char*));
    if (img->pixels == NULL) { printf("ERROR: Memory allocation failed for row pointers.\n"); fclose(fp); return 0; }
    
    // P5 (binary) reading
    if (strcmp(magic, "P5") == 0) {
        for (int i = 0; i < img->height; i++) {
            img->pixels[i] = (unsigned char*)malloc(img->width * sizeof(unsigned char));
            if (img->pixels[i] == NULL) {
                 for (int j = 0; j < i; j++) free(img->pixels[j]);
                 free(img->pixels); img->pixels = NULL; fclose(fp); return 0;
            }
            if (fread(img->pixels[i], sizeof(unsigned char), img->width, fp) != img->width) {
                 printf("ERROR: Reading pixel data failed for P5 row %d.\n", i);
                 for (int j = 0; j <= i; j++) free(img->pixels[j]);
                 free(img->pixels); img->pixels = NULL; fclose(fp); return 0;
            }
        }
    } 
    // P2 (ASCII) reading
    else if (strcmp(magic, "P2") == 0) {
        for (int i = 0; i < img->height; i++) {
            img->pixels[i] = (unsigned char*)malloc(img->width * sizeof(unsigned char));
            if (img->pixels[i] == NULL) {
                 for (int j = 0; j < i; j++) free(img->pixels[j]);
                 free(img->pixels); img->pixels = NULL; fclose(fp); return 0;
            }
            for (int j = 0; j < img->width; j++) {
                int val;
                if (fscanf(fp, "%d", &val) != 1) {
                    printf("ERROR: Reading pixel data failed for P2 pixel [%d][%d].\n", i, j);
                    for (int k = 0; k <= i; k++) free(img->pixels[k]);
                    free(img->pixels); fclose(fp); return 0;
                }
                img->pixels[i][j] = (unsigned char)val;
            }
        }
    }
    
    fclose(fp);
    printf("SUCCESS: Image '%s' loaded. Format: %s. Dimensions: %d x %d (Max Val: %d)\n", 
           filename, magic, img->width, img->height, img->max_val);
    return 1;
}

// 6. Save the  PGM File
int save_pgm_image(const PGMImage* img, const char* filename) {
    if (!is_image_loaded(img)) return 0;
    FILE* fp = fopen(filename, "wb"); 
    if (fp == NULL) {
        perror("Error creating file"); return 0;
    }
    fprintf(fp, "P5\n");
    fprintf(fp, "%d %d\n", img->width, img->height);
    fprintf(fp, "%d\n", img->max_val);
    for (int i = 0; i < img->height; i++) {
        if (fwrite(img->pixels[i], sizeof(unsigned char), img->width, fp) != img->width) {
            printf("ERROR: Writing pixel data failed for row %d.\n", i); fclose(fp); return 0;
        }
    }
    fclose(fp);
    printf("SUCCESS: Image saved to '%s'.\n", filename);
    return 1;
}

// 2. Zoom or Shrink the Image 

void nearest_neighbor_zoom(const PGMImage* original, PGMImage* new_img, int factor) {
    for (int i_new = 0; i_new < new_img->height; i_new++) {
        for (int j_new = 0; j_new < new_img->width; j_new++) {
            int i_orig = i_new / factor;
            int j_orig = j_new / factor;
            new_img->pixels[i_new][j_new] = original->pixels[i_orig][j_orig];
        }
    }
}

void subsample_shrink(const PGMImage* original, PGMImage* new_img, int factor) {
    for (int i_new = 0; i_new < new_img->height; i_new++) {
        for (int j_new = 0; j_new < new_img->width; j_new++) {
            int i_orig = i_new * factor;
            int j_orig = j_new * factor;
            new_img->pixels[i_new][j_new] = original->pixels[i_orig][j_orig];
        }
    }
}

void resize_image(PGMImage* current_img) {
    char input_factor[10];
    printf("Enter scaling factor (e.g., 2 for 2x, 0.5 for 0.5x): ");
    scanf("%s", input_factor);
    PGMImage new_image = {0, 0, 0, NULL}; 
    int w = current_img->width;
    int h = current_img->height;
    int factor = 0;

    if (strcmp(input_factor, "2") == 0 || strcmp(input_factor, "3") == 0) {
        factor = atoi(input_factor);
        create_new_image(current_img, &new_image, w * factor, h * factor);
        nearest_neighbor_zoom(current_img, &new_image, factor);
        printf("SUCCESS: Image zoomed by %dx.\n", factor);
    } else if (strcmp(input_factor, "0.5") == 0) {
        factor = 2; 
        if (w % factor != 0 || h % factor != 0) {
             printf("ERROR: Dimensions must be divisible by 2 for shrinking.\n"); return;
        }
        create_new_image(current_img, &new_image, w / factor, h / factor);
        subsample_shrink(current_img, &new_image, factor);
        printf("SUCCESS: Image shrunk by 0.5x.\n");
    } else if (strcmp(input_factor, "0.25") == 0) {
        factor = 4; 
        if (w % factor != 0 || h % factor != 0) {
             printf("ERROR: Dimensions must be divisible by 4 for shrinking.\n"); return;
        }
        create_new_image(current_img, &new_image, w / factor, h / factor);
        subsample_shrink(current_img, &new_image, factor);
        printf("SUCCESS: Image shrunk by 0.25x.\n");
    } else {
        printf("ERROR: Invalid scaling factor entered. Supported: 2, 3, 0.5, 0.25.\n");
        return;
    }
    free_image_memory(current_img);
    *current_img = new_image; 
}

// Apply Filters

void average_filter(const PGMImage* original, PGMImage* new_img) {
    deep_copy_image(original, new_img); 
    for (int i = 1; i < new_img->height - 1; i++) {
        for (int j = 1; j < new_img->width - 1; j++) {
            long sum = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    sum += original->pixels[i + k][j + l];
                }
            }
            new_img->pixels[i][j] = (unsigned char)(sum / 9);
        }
    }
}

void sort_nine(unsigned char arr[9]) {
    for (int i = 0; i < 8; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (arr[i] > arr[j]) {
                unsigned char temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
}

void median_filter(const PGMImage* original, PGMImage* new_img) {
    deep_copy_image(original, new_img);
    unsigned char window[9];

    for (int i = 1; i < new_img->height - 1; i++) {
        for (int j = 1; j < new_img->width - 1; j++) {
            int k = 0;
            for (int row = -1; row <= 1; row++) {
                for (int col = -1; col <= 1; col++) {
                    window[k++] = original->pixels[i + row][j + col];
                }
            }
            sort_nine(window);
            new_img->pixels[i][j] = window[4]; 
        }
    }
}

void apply_filter(PGMImage* current_img) {
    int filter_choice;
    printf("1 - Apply Average/Mean Filter (3x3)\n");
    printf("2 - Apply Median Filter (3x3)\n");
    printf("Enter filter choice: ");
    if (scanf("%d", &filter_choice) != 1) { 
        printf("Invalid input.\n"); while(getchar() != '\n'); return; 
    }

    PGMImage new_image = {0, 0, 0, NULL};

    switch (filter_choice) {
        case 1:
            average_filter(current_img, &new_image);
            printf("SUCCESS: Average (Mean) filter applied.\n");
            break;
        case 2:
            median_filter(current_img, &new_image);
            printf("SUCCESS: Median filter applied.\n");
            break;
        default:
            printf("Invalid filter choice.\n");
            return;
    }

    free_image_memory(current_img);
    *current_img = new_image;
}

// Edge Detection

void sobel_edge_detection(const PGMImage* original, PGMImage* new_img) {
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    create_new_image(original, new_img, original->width, original->height);
    for (int i = 1; i < original->height - 1; i++) {
        for (int j = 1; j < original->width - 1; j++) {
            long gx_sum = 0;
            long gy_sum = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    int pixel_val = original->pixels[i + k][j + l];
                    gx_sum += pixel_val * Gx[k + 1][l + 1];
                    gy_sum += pixel_val * Gy[k + 1][l + 1];
                }
            }
            long magnitude = labs(gx_sum) + labs(gy_sum);
            unsigned char final_value = (unsigned char)(magnitude > 255 ? 255 : magnitude);
            new_img->pixels[i][j] = final_value;
        }
    }
}

void prewitt_edge_detection(const PGMImage* original, PGMImage* new_img) {
    const int Gx[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    create_new_image(original, new_img, original->width, original->height);
    for (int i = 1; i < original->height - 1; i++) {
        for (int j = 1; j < original->width - 1; j++) {
            long gx_sum = 0;
            long gy_sum = 0;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    int pixel_val = original->pixels[i + k][j + l];
                    gx_sum += pixel_val * Gx[k + 1][l + 1];
                    gy_sum += pixel_val * Gy[k + 1][l + 1];
                }
            }
            long magnitude = labs(gx_sum) + labs(gy_sum);
            unsigned char final_value = (unsigned char)(magnitude > 255 ? 255 : magnitude);
            new_img->pixels[i][j] = final_value;
        }
    }
}

// Canny Edge Detector

void gaussian_blur(const PGMImage* original, float** blurred) {
    const int kernel[5][5] = {
        {2, 4, 5, 4, 2},
        {4, 9, 12, 9, 4},
        {5, 12, 15, 12, 5},
        {4, 9, 12, 9, 4},
        {2, 4, 5, 4, 2}
    };
    int kernel_sum = 159; 

    int W = original->width;
    int H = original->height;

    for (int i = 2; i < H - 2; i++) {
        for (int j = 2; j < W - 2; j++) {
            long sum = 0;
            for (int k = -2; k <= 2; k++) {
                for (int l = -2; l <= 2; l++) {
                    sum += original->pixels[i + k][j + l] * kernel[k + 2][l + 2];
                }
            }
            blurred[i][j] = (float)sum / kernel_sum;
        }
    }
}

void compute_gradient_and_magnitude(float** blurred, int W, int H, float** magnitude, float** angle) {
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    float max_mag = 0.0f;

    for (int i = 1; i < H - 1; i++) {
        for (int j = 1; j < W - 1; j++) {
            float gx_sum = 0.0f; 
            float gy_sum = 0.0f; 

            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    float pixel_val = blurred[i + k][j + l];
                    gx_sum += pixel_val * Gx[k + 1][l + 1];
                    gy_sum += pixel_val * Gy[k + 1][l + 1];
                }
            }

            magnitude[i][j] = sqrtf(gx_sum * gx_sum + gy_sum * gy_sum);
            angle[i][j] = atan2f(gy_sum, gx_sum) * 180.0f / PI;

            if (magnitude[i][j] > max_mag) max_mag = magnitude[i][j];
        }
    }
}

void non_maximum_suppression(float** mag, float** angle, int W, int H, unsigned char** suppressed) {
    for (int i = 1; i < H - 1; i++) {
        for (int j = 1; j < W - 1; j++) {
            float q = 255.0f, r = 255.0f; 
            float current_angle = angle[i][j];

            if (current_angle < 0) current_angle += 180.0f;

            if ((current_angle >= 0 && current_angle < 22.5) || (current_angle >= 157.5 && current_angle <= 180)) {
                q = mag[i][j + 1];
                r = mag[i][j - 1];
            } else if (current_angle >= 22.5 && current_angle < 67.5) {
                q = mag[i - 1][j + 1];
                r = mag[i + 1][j - 1];
            } else if (current_angle >= 67.5 && current_angle < 112.5) {
                q = mag[i - 1][j];
                r = mag[i + 1][j];
            } else if (current_angle >= 112.5 && current_angle < 157.5) {
                q = mag[i - 1][j - 1];
                r = mag[i + 1][j + 1];
            }

            if (mag[i][j] >= q && mag[i][j] >= r) {
                suppressed[i][j] = (unsigned char)fminf(mag[i][j], 255.0f);
            } else {
                suppressed[i][j] = 0;
            }
        }
    }
}

void hysteresis_thresholding(unsigned char** suppressed, int W, int H, unsigned char** final_edges) {
    int max_val = 0;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (suppressed[i][j] > max_val) max_val = suppressed[i][j];
        }
    }

    int high_thresh = (int)(max_val * HIGH_THRESHOLD_RATIO);
    int low_thresh = (int)(max_val * LOW_THRESHOLD_RATIO);
    
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (suppressed[i][j] >= high_thresh) {
                final_edges[i][j] = 255; 
            } else if (suppressed[i][j] >= low_thresh) {
                final_edges[i][j] = 100; // Weak edge
            } else {
                final_edges[i][j] = 0; 
            }
        }
    }

    // Edge Tracking 
    for (int i = 1; i < H - 1; i++) {
        for (int j = 1; j < W - 1; j++) {
            if (final_edges[i][j] == 255) {
                edge_tracking(final_edges, i, j, W, H, 255);
            }
        }
    }
    
    // suppress all remaining weak edges (100)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            if (final_edges[i][j] == 100) {
                final_edges[i][j] = 0;
            }
        }
    }
}

void edge_tracking(unsigned char** edges, int x, int y, int W, int H, int high_thresh) {
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            
            int nx = x + i; 
            int ny = y + j; 

            if (nx > 0 && nx < H - 1 && ny > 0 && ny < W - 1) {
                if (edges[nx][ny] == 100) { 
                    edges[nx][ny] = high_thresh; 
                    edge_tracking(edges, nx, ny, W, H, high_thresh); 
                }
            }
        }
    }
}

void canny_edge_detector(PGMImage* current_img) {
    if (!is_image_loaded(current_img)) return;

    int W = current_img->width;
    int H = current_img->height;

    // Allocate memory for intermediate steps
    float** blurred = create_float_array(H, W);
    float** magnitude = create_float_array(H, W);
    float** angle = create_float_array(H, W);
    
    if (!blurred || !magnitude || !angle) {
        printf("ERROR: Memory allocation failed for Canny intermediate arrays.\n");
        free_float_array(blurred, H);
        free_float_array(magnitude, H);
        free_float_array(angle, H);
        return;
    }

    // Gaussian Smoothing 
    gaussian_blur(current_img, blurred); 

    // Compute Gradient and Magnitude
    compute_gradient_and_magnitude(blurred, W, H, magnitude, angle);

    // Non-Maximum Suppression
    PGMImage temp_img = {0, 0, 0, NULL};
    create_new_image(current_img, &temp_img, W, H);
    non_maximum_suppression(magnitude, angle, W, H, temp_img.pixels);

    // Thresholding
    PGMImage final_img = {0, 0, 0, NULL};
    create_new_image(current_img, &final_img, W, H);
    hysteresis_thresholding(temp_img.pixels, W, H, final_img.pixels);

    // Free intermediate memory
    free_float_array(blurred, H);
    free_float_array(magnitude, H);
    free_float_array(angle, H);
    free_image_memory(&temp_img); 

    // Replace the old image with the final Canny result
    free_image_memory(current_img);
    *current_img = final_img;
    printf("SUCCESS: Canny Edge Detector (4-Stage) applied.\n");
}


void edge_detection(PGMImage* current_img) {
    int edge_choice;
    printf("1 - Apply Sobel Edge Filter\n");
    printf("2 - Apply Prewitt Edge Filter\n");
    printf("3 - Apply Canny Edge Detector (Complete)\n");
    printf("Enter edge detection choice: ");
    if (scanf("%d", &edge_choice) != 1) { 
        printf("Invalid input.\n"); while(getchar() != '\n'); return; 
    }

    PGMImage new_image = {0, 0, 0, NULL};

    switch (edge_choice) {
        case 1:
            sobel_edge_detection(current_img, &new_image);
            printf("SUCCESS: Sobel Edge Detector applied.\n");
            break;
        case 2:
            prewitt_edge_detection(current_img, &new_image);
            printf("SUCCESS: Prewitt Edge Filter applied.\n");
            break;
        case 3:
            canny_edge_detector(current_img);
            return;
        default:
            printf("Invalid edge detection choice.\n");
            return;
    }

    free_image_memory(current_img);
    *current_img = new_image;
}

// Compute Local Binary Pattern (LBP)

void calculate_lbp(const PGMImage* original, PGMImage* new_img) {
    create_new_image(original, new_img, original->width, original->height);
    const int offsets[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, 
                             {1, 1}, {1, 0}, {1, -1}, {0, -1}};

    for (int i = 1; i < original->height - 1; i++) {
        for (int j = 1; j < original->width - 1; j++) {
            unsigned char center = original->pixels[i][j];
            int lbp_code = 0;

            for (int k = 0; k < 8; k++) {
                int ni = i + offsets[k][0]; 
                int nj = j + offsets[k][1]; 
                unsigned char neighbor = original->pixels[ni][nj];

                if (neighbor >= center) {
                    lbp_code |= (1 << (7 - k));
                }
            }
            new_img->pixels[i][j] = (unsigned char)lbp_code;
        }
    }
}

void compute_lbp(PGMImage* current_img) {
    PGMImage new_image = {0, 0, 0, NULL};
    calculate_lbp(current_img, &new_image);
    printf("SUCCESS: Local Binary Pattern (LBP) calculated.\n");

    free_image_memory(current_img);
    *current_img = new_image;
}