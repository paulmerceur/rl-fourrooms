// Full implementation copied to be standalone like the Ocean version
#include <stdlib.h>
#include <string.h>

#ifndef NO_RAYLIB
#include "raylib.h"
#else
typedef struct { unsigned char r, g, b, a; } Color;
typedef struct { int id; } Texture2D;
#define WHITE (Color){255, 255, 255, 255}
#endif

// Action space
const unsigned char LEFT = 0;
const unsigned char RIGHT = 1; 
const unsigned char FORWARD = 2;
const unsigned char PICKUP = 3; // Unused
const unsigned char DROP = 4; // Unused
const unsigned char TOGGLE = 5; // Unused
const unsigned char DONE = 6; // Unused

// Observation: Objects
const unsigned char UNSEEN = 0;
const unsigned char EMPTY = 1;
const unsigned char WALL = 2;
const unsigned char FLOOR = 3; // Unused
const unsigned char DOOR = 4; // Unused
const unsigned char KEY = 5; // Unused
const unsigned char BALL = 6; // Unused
const unsigned char BOX = 7; // Unused
const unsigned char GOAL = 8;
const unsigned char LAVA = 9; // Unused
const unsigned char AGENT = 10;

// Observation: Colors
const unsigned char COLOR_BLACK = 0;
const unsigned char COLOR_GREEN = 1;
const unsigned char COLOR_BLUE = 2;
const unsigned char COLOR_PURPLE = 3;
const unsigned char COLOR_YELLOW = 4;
const unsigned char COLOR_GREY = 5;

// PufferLib standard colors for rendering
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_BACKGROUND2 = (Color){18, 72, 72, 255};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Log log;
    unsigned char* observations; // 7x7x3 observation: (OBJECT_IDX, COLOR_IDX, STATE) per cell
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int size; // default 19
    int tick;
    int agent_x, agent_y;
    int agent_dir; // 0=East, 1=South, 2=West, 3=North
    int goal_x, goal_y;
    unsigned char* grid; // Stores OBJECT_IDX values
    int see_through_walls;
    Texture2D puffers;
} FourRooms;

void add_log(FourRooms* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1.0 : 0.0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

int can_see_cell(FourRooms* env, int agent_x, int agent_y, int target_x, int target_y) {
    if (env->see_through_walls) {
        return 1;
    }

    // Use Bresenham's line algorithm to check line of sight
    int dx = abs(target_x - agent_x);
    int dy = abs(target_y - agent_y);
    int x = agent_x;
    int y = agent_y;
    int x_inc = (target_x > agent_x) ? 1 : -1;
    int y_inc = (target_y > agent_y) ? 1 : -1;
    int error = dx - dy;

    while (x != target_x || y != target_y) {
        // If we've reached the target cell, stop (target cell should always be visible)
        if (x == target_x && y == target_y) {
            break;
        }

        int error2 = 2 * error;
        if (error2 > -dy) {
            error -= dy;
            x += x_inc;
        }
        if (error2 < dx) {
            error += dx;
            y += y_inc;
        }

        // If the next cell (not the target) is a wall, block vision beyond but allow seeing the wall itself
        if ((x != target_x || y != target_y) &&
            x >= 0 && x < env->size && y >= 0 && y < env->size &&
            env->grid[y * env->size + x] == WALL) {
            return 0; // Wall blocks the view beyond, but wall itself is visible
        }
    }
    return 1; // Target cell is visible
}

void generate_observation(FourRooms* env) {
    // Generate 7x7x3 observation centered on agent's view direction
    int view_size = 7;
    int half_view = view_size / 2;
    
    // Calculate the center of the view based on agent's direction
    int center_x = env->agent_x;
    int center_y = env->agent_y;
    
    // Shift center forward in the direction the agent is facing
    if (env->agent_dir == 0) center_x += half_view; // East
    else if (env->agent_dir == 1) center_y += half_view; // South
    else if (env->agent_dir == 2) center_x -= half_view; // West
    else if (env->agent_dir == 3) center_y -= half_view; // North
    
    for (int i = 0; i < view_size; i++) {
        for (int j = 0; j < view_size; j++) {
            int world_x = center_x - half_view + j;
            int world_y = center_y - half_view + i;
            
            // Calculate flat index for this cell in the 7x7x3 observation
            int base_idx = (i * view_size + j) * 3;
            
            unsigned char object_idx, color_idx, state;
            
            // Check bounds, out of bounds is treated as wall
            if (world_x < 0 || world_x >= env->size || world_y < 0 || world_y >= env->size) {
                object_idx = WALL;
                color_idx = COLOR_GREY;
                state = 0;
            } else if (!can_see_cell(env, env->agent_x, env->agent_y, world_x, world_y)) {
                object_idx = UNSEEN; // Cell is blocked by walls
                color_idx = COLOR_BLACK;
                state = 0;
            } else {
                int grid_idx = world_y * env->size + world_x;
                unsigned char grid_cell = env->grid[grid_idx];
                
                // Map grid cell to MiniGrid encoding
                switch (grid_cell) {
                    case EMPTY:
                        object_idx = EMPTY;
                        color_idx = COLOR_BLACK;
                        state = 0;
                        break;
                    case WALL:
                        object_idx = WALL;
                        color_idx = COLOR_GREY;
                        state = 0;
                        break;
                    case AGENT:
                        object_idx = AGENT;
                        color_idx = COLOR_BLUE;
                        state = 0;
                        break;
                    case GOAL:
                        object_idx = GOAL;
                        color_idx = COLOR_GREEN;
                        state = 0;
                        break;
                    default:
                        object_idx = EMPTY;
                        color_idx = 0;
                        state = 0;
                        break;
                }
            }
            
            env->observations[base_idx] = object_idx;
            env->observations[base_idx + 1] = color_idx;
            env->observations[base_idx + 2] = state;
        }
    }
}

void create_four_rooms_grid(FourRooms* env) {
    int size = env->size;
    
    // Clear grid
    memset(env->grid, EMPTY, size * size * sizeof(unsigned char));
    
    // Create outer walls
    for (int i = 0; i < size; i++) {
        env->grid[0 * size + i] = WALL; // Top
        env->grid[(size-1) * size + i] = WALL; // Bottom
        env->grid[i * size + 0] = WALL; // Left
        env->grid[i * size + (size-1)] = WALL; // Right
    }
    
    int room_w = size / 2;
    int room_h = size / 2;
    
    // Create vertical separating wall
    for (int y = 0; y < size; y++) {
        env->grid[y * size + room_w] = WALL;
    }
    
    // Create horizontal separating wall
    for (int x = 0; x < size; x++) {
        env->grid[room_h * size + x] = WALL;
    }
    
    // Create 4 gaps in the separating walls
    // Gap in vertical wall (top half)
    int gap_y1 = 1 + rand() % (room_h - 2);
    env->grid[gap_y1 * size + room_w] = EMPTY;
    
    // Gap in vertical wall (bottom half)
    int gap_y2 = room_h + 1 + rand() % (room_h - 2);
    env->grid[gap_y2 * size + room_w] = EMPTY;
    
    // Gap in horizontal wall (left half)
    int gap_x1 = 1 + rand() % (room_w - 2);
    env->grid[room_h * size + gap_x1] = EMPTY;
    
    // Gap in horizontal wall (right half)
    int gap_x2 = room_w + 1 + rand() % (room_w - 2);
    env->grid[room_h * size + gap_x2] = EMPTY;
}

void c_reset(FourRooms* env) {

    create_four_rooms_grid(env);
    
    // Place agent randomly in valid position
    do {
        env->agent_x = 1 + rand() % (env->size - 2);
        env->agent_y = 1 + rand() % (env->size - 2);
    } while (env->grid[env->agent_y * env->size + env->agent_x] != EMPTY);
    
    // Place goal randomly in valid position (different from agent)
    do {
        env->goal_x = 1 + rand() % (env->size - 2);
        env->goal_y = 1 + rand() % (env->size - 2);
    } while (env->grid[env->goal_y * env->size + env->goal_x] != EMPTY ||
             (env->goal_x == env->agent_x && env->goal_y == env->agent_y));
    
    // Set agent and goal on grid
    env->grid[env->agent_y * env->size + env->agent_x] = AGENT;
    env->grid[env->goal_y * env->size + env->goal_x] = GOAL;
    
    // Random initial direction
    env->agent_dir = rand() % 4;
    env->tick = 0;
    
    generate_observation(env);
}

void c_step(FourRooms* env) {
    env->tick += 1;
    
    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;
    
    // Clear agent from current position
    env->grid[env->agent_y * env->size + env->agent_x] = EMPTY;
    
    int new_x = env->agent_x;
    int new_y = env->agent_y;
    int new_dir = env->agent_dir;
    
    if (action == LEFT) {
        new_dir = (env->agent_dir + 3) % 4;
    } else if (action == RIGHT) {
        new_dir = (env->agent_dir + 1) % 4;
    } else if (action == FORWARD) {
        if (env->agent_dir == 0) new_x += 1;
        else if (env->agent_dir == 1) new_y += 1;
        else if (env->agent_dir == 2) new_x -= 1;
        else if (env->agent_dir == 3) new_y -= 1;

        // Check if move is valid
        if (new_x >= 0 && new_x < env->size && new_y >= 0 && new_y < env->size &&
            env->grid[new_y * env->size + new_x] != WALL) {
            env->agent_x = new_x;
            env->agent_y = new_y;
        }
    }
    
    env->agent_dir = new_dir;
    
    // Check if agent reached goal
    if (env->agent_x == env->goal_x && env->agent_y == env->goal_y) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0;
        add_log(env);
        c_reset(env);
        return;
    }
    
    // Place agent back on grid
    env->grid[env->agent_y * env->size + env->agent_x] = AGENT;
    
    // Check timeout
    if (env->tick >= env->size * env->size) {
        env->terminals[0] = 1;
        env->rewards[0] = 0.0;
        add_log(env);
        c_reset(env);
        return;
    }
    
    generate_observation(env);
}

#ifndef NO_RAYLIB
void c_render(FourRooms* env) {
    if (!IsWindowReady()) {
        InitWindow(32*env->size, 32*env->size, "PufferLib FourRooms");
        SetTargetFPS(10);
        env->puffers = LoadTexture("resources/shared/puffers_128.png");
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    int px = 32;
    
    // Draw the main grid
    for (int y = 0; y < env->size; y++) {
        for (int x = 0; x < env->size; x++) {
            int cell = env->grid[y * env->size + x];
            Color color = PUFF_BACKGROUND;
            
            if (cell == WALL) color = PUFF_BACKGROUND2;
            else if (cell == GOAL) color = PUFF_RED;

            if (cell != EMPTY && cell != AGENT) {
                DrawRectangle(x*px, y*px, px, px, color);
            }
        }
    }
    
    // Draw agent's 7x7 observation window
    int view_size = 7;
    int half_view = view_size / 2;
    
    // Calculate the center of the view based on agent's direction
    int center_x = env->agent_x;
    int center_y = env->agent_y;
    
    // Shift center forward in the direction the agent is facing
    if (env->agent_dir == 0) center_x += half_view; // East
    else if (env->agent_dir == 1) center_y += half_view; // South
    else if (env->agent_dir == 2) center_x -= half_view; // West
    else if (env->agent_dir == 3) center_y -= half_view; // North
    
    // Draw semi-transparent overlay for observation window
    Color obs_overlay = (Color){180, 180, 180, 80};
    for (int i = 0; i < view_size; i++) {
        for (int j = 0; j < view_size; j++) {
            int world_x = center_x - half_view + j;
            int world_y = center_y - half_view + i;
            
            // Only draw overlay for cells within grid bounds and visible to agent
            if (world_x >= 0 && world_x < env->size && world_y >= 0 && world_y < env->size &&
                can_see_cell(env, env->agent_x, env->agent_y, world_x, world_y)) {
                DrawRectangle(world_x*px, world_y*px, px, px, obs_overlay);
            }
        }
    }
    
    // Draw agent
    int starting_sprite_x = 0;
    int rotation = 90 * env->agent_dir; // 0=East(0°), 1=South(90°), 2=West(180°), 3=North(270°)
    if (rotation == 180) {
        starting_sprite_x = 128; // Use flipped sprite for 180° rotation
        rotation = 0;
    }
    
    DrawTexturePro(
        env->puffers,
        (Rectangle){starting_sprite_x, 0, 128, 128},
        (Rectangle){
            env->agent_x * px + px/2,
            env->agent_y * px + px/2,
            px,
            px
        },
        (Vector2){px/2, px/2},
        rotation,
        WHITE
    );

    EndDrawing();
}

void c_close(FourRooms* env) {
    if (IsWindowReady()) {
        UnloadTexture(env->puffers);
        CloseWindow();
    }
    if (env->grid) {
        free(env->grid);
    }
}
#else
void c_render(FourRooms* env) { (void)env; }
void c_close(FourRooms* env) {
    if (env->grid) { free(env->grid); }
}
#endif




