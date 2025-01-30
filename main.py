import pygame
import numpy as np
from perlin_noise import PerlinNoise
from gan_integration import generate_grass_texture  # Импортируем генерацию текстуры травы через GAN

# Параметры игры
WINDOW_WIDTH, WINDOW_HEIGHT = 1920, 1080
TILE_SIZE = 15
LEVEL_WIDTH, LEVEL_HEIGHT = 100, 100  # Размер чанка карты
SCALE = 30.0
PLAYER_SIZE = TILE_SIZE
PLAYER_SPEED = 5
SPRINT_SPEED = 20
CHUNK_SIZE = 100  # Размер чанка

# Генерация уровня с помощью шума Перлина
def generate_level(width, height, scale=30.0):
    noise = PerlinNoise(octaves=4)
    level = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            level[x, y] = noise([x / scale, y / scale])
    return level

def create_semantic_level(level):
    """Создание семантической карты с уменьшением плотности травы."""
    semantic_level = np.zeros(level.shape)
    
    # Трава  (значения шума < -0.2) - доступна для движения
    semantic_level[level < -0.1] = 1
    
    # Почва (значения шума от -0.2 до 0.1) - недоступна для движения
    semantic_level[(level >= -0.1) & (level < 0.01)] = 2 
    
    # Трава  (значения шума >= 0.1) - удаляем траву на высоких уровнях шума
    semantic_level[level >= 0.01] = 1
    
    # Дополнительная логика для контроля плотности травы
    # Снижение плотности травы в верхних частях карты
    high_terrain_threshold = 0.3
    semantic_level[level > high_terrain_threshold] = 1  # Трава  на высоких уровнях

    

    return semantic_level


def get_chunk(x, y, level_cache):
    """Получить чанки уровня вокруг текущей позиции."""
    chunks = {}
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            if (i, j) not in level_cache:
                level_cache[(i, j)] = create_semantic_level(generate_level(CHUNK_SIZE, CHUNK_SIZE, scale=SCALE))
            chunks[(i, j)] = level_cache[(i, j)]
    return chunks

def find_safe_start_position(level_chunks, chunk_x, chunk_y):
    """Находит безопасное место для старта игрока на почве (не на траве)."""
    for x in range(CHUNK_SIZE):
        for y in range(CHUNK_SIZE):
            if int(level_chunks[(chunk_x, chunk_y)][x, y]) == 1:  # Почва
                return (chunk_x * CHUNK_SIZE + x) * TILE_SIZE, (chunk_y * CHUNK_SIZE + y) * TILE_SIZE

    # Если в текущем чанке нет безопасного места, ищем дальше
    for key, level in level_chunks.items():
        for x in range(CHUNK_SIZE):
            for y in range(CHUNK_SIZE):
                if int(level[x, y]) == 1:
                    return (key[0] * CHUNK_SIZE + x) * TILE_SIZE, (key[1] * CHUNK_SIZE + y) * TILE_SIZE
    return 0, 0  # Если вообще не нашлось места

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()

# Загрузка текстуры травы через GAN
grass_texture_data = generate_grass_texture()
grass_texture = pygame.image.frombuffer(grass_texture_data.tobytes(), (64, 64), 'RGB')
grass_texture = pygame.transform.scale(grass_texture, (TILE_SIZE, TILE_SIZE))

# Инициализация игрока
player_image = pygame.Surface((PLAYER_SIZE, PLAYER_SIZE))
player_image.fill((255, 0, 0))  # Красный цвет

# Отрисовка уровня
def draw_level(level_chunks, player_x, player_y):
    """Отрисовка уровня на экране."""
    for (chunk_x, chunk_y), level in level_chunks.items():
        for x in range(CHUNK_SIZE):
            for y in range(CHUNK_SIZE):
                tile_type = int(level[x, y])
                if tile_type == 2:
                    # Почва 
                    color = (0, 0, 0)
                    pygame.draw.rect(screen, color, pygame.Rect(
                        (chunk_x * CHUNK_SIZE + x) * TILE_SIZE - player_x,
                        (chunk_y * CHUNK_SIZE + y) * TILE_SIZE - player_y,
                        TILE_SIZE,
                        TILE_SIZE
                    ))
                elif tile_type == 1:
                    # Трава (нельзя заходить, отображаем текстуру)
                    screen.blit(grass_texture, pygame.Rect(
                        (chunk_x * CHUNK_SIZE + x) * TILE_SIZE - player_x,
                        (chunk_y * CHUNK_SIZE + y) * TILE_SIZE - player_y,
                        TILE_SIZE,
                        TILE_SIZE
                    ))

# Основной игровой цикл
def main():
    chunk_x = 0
    chunk_y = 0

    # Инициализация кэша чанков уровня
    level_cache = {}
    level_chunks = get_chunk(chunk_x, chunk_y, level_cache)
    camera_x, camera_y = 0, 0
    # Найти безопасное место для начала
    player_x, player_y = find_safe_start_position(level_chunks, chunk_x, chunk_y)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Определяем текущую скорость
        current_speed = PLAYER_SPEED
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:  # Проверяем, зажат ли Shift
            current_speed = SPRINT_SPEED

        new_x, new_y = player_x, player_y
        if keys[pygame.K_a]:
            new_x -= current_speed
        if keys[pygame.K_d]:
            new_x += current_speed
        if keys[pygame.K_w]:
            new_y -= current_speed
        if keys[pygame.K_s]:
            new_y += current_speed

        # Определяем чанки вокруг персонажа
        new_chunk_x = new_x // (CHUNK_SIZE * TILE_SIZE)
        new_chunk_y = new_y // (CHUNK_SIZE * TILE_SIZE)

        if (new_chunk_x, new_chunk_y) != (chunk_x, chunk_y):
            chunk_x, chunk_y = new_chunk_x, new_chunk_y
            level_chunks = get_chunk(chunk_x, chunk_y, level_cache)

        # Проверка на прохождение через траву (нельзя ходить по траве)
        new_x_chunk = new_x // TILE_SIZE
        new_y_chunk = new_y // TILE_SIZE
        if int(level_chunks[(new_x // (CHUNK_SIZE * TILE_SIZE), new_y // (CHUNK_SIZE * TILE_SIZE))]
                   [new_x_chunk % CHUNK_SIZE, new_y_chunk % CHUNK_SIZE]) == 1:  # Почва
            player_x, player_y = new_x, new_y

        # Плавное движение камеры
        camera_x += (player_x - WINDOW_WIDTH // 2 - camera_x) * 0.1
        camera_y += (player_y - WINDOW_HEIGHT // 2 - camera_y) * 0.1

        screen.fill((0, 0, 0))  # Фон
        draw_level(level_chunks, camera_x, camera_y)
        screen.blit(player_image, (WINDOW_WIDTH // 2 - PLAYER_SIZE // 2, WINDOW_HEIGHT // 2 - PLAYER_SIZE // 2))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
