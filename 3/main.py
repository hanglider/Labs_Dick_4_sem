import tkinter as tk
from tkinter import messagebox, simpledialog
import random
import math
import time

class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Задача коммивояжёра (метод ближайшего соседа)")

        # ---------- Основные фреймы ----------
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        top_frame = tk.Frame(main_frame)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---------- Верхний Canvas: редактирование графа ----------
        self.canvas_top = tk.Canvas(top_frame, width=600, height=400, bg="white")
        self.canvas_top.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_top.bind("<Button-1>", self.on_canvas_top_click)

        # Боковая панель (справа от верхнего canvas)
        side_panel = tk.Frame(top_frame, padx=5, pady=5)
        side_panel.pack(side=tk.LEFT, fill=tk.Y)

        # Режимы (вершины/рёбра)
        self.mode = tk.StringVar(value="vertex")  # "vertex" или "edge"
        tk.Radiobutton(side_panel, text="Добавлять вершины",
                       variable=self.mode, value="vertex").pack(anchor=tk.W)
        tk.Radiobutton(side_panel, text="Добавлять рёбра",
                       variable=self.mode, value="edge").pack(anchor=tk.W)

        # Направление
        # tk.Label(side_panel, text="Направление:").pack(anchor=tk.W)
        # self.direction_var = tk.StringVar(value="forward")
        # tk.Radiobutton(side_panel, text="1 → 2",
        #                variable=self.direction_var, value="forward").pack(anchor=tk.W)
        # tk.Radiobutton(side_panel, text="2 → 1",
        #                variable=self.direction_var, value="backward").pack(anchor=tk.W)
        # tk.Radiobutton(side_panel, text="Обе (разные веса)",
        #                variable=self.direction_var, value="both").pack(anchor=tk.W)

        # Поле для ввода веса (основного)
        tk.Label(side_panel, text="Вес (для одного направления):").pack(anchor=tk.W)
        self.weight_entry = tk.Entry(side_panel, width=5)
        self.weight_entry.pack(anchor=tk.W)
        self.weight_entry.insert(0, "1")  # по умолчанию 1

        # Кнопка «Запустить TSP»
        self.tsp_button = tk.Button(side_panel, text="Запустить TSP", command=self.run_tsp)
        self.tsp_button.pack(pady=5)

        # Новая кнопка: Запустить модифицированный TSP
        self.tsp_modified_button = tk.Button(side_panel, text="Запустить модифицированный TSP", command=self.run_tsp_modified)
        self.tsp_modified_button.pack(pady=5)

        # Кнопка «Имитация отжига»
        self.annealing_button = tk.Button(side_panel, text="Имитация отжига", command=self.run_simulated_annealing)
        self.annealing_button.pack(pady=5)

        # Кнопка «Сверхбыстрый отжиг»
        self.fast_annealing_button = tk.Button(side_panel, text="Сверхбыстрый отжиг", command=self.run_fast_simulated_annealing)
        self.fast_annealing_button.pack(pady=5)

        # Кнопка: Муравьиный алгоритм
        self.aco_button = tk.Button(side_panel, text="Муравьиный алгоритм", command=self.run_ant_colony)
        self.aco_button.pack(pady=5)

        # Новая кнопка: Элитные муравьи
        self.elite_aco_button = tk.Button(side_panel, text="Элитные муравьи", command=self.run_elite_ant_colony)
        self.elite_aco_button.pack(pady=5)

        # Кнопка «Удалить вершину»
        self.delete_vertex_button = tk.Button(side_panel, text="Удалить вершину", command=self.delete_vertex_dialog)
        self.delete_vertex_button.pack(pady=5)

        # Кнопка «Удалить ребро»
        self.delete_edge_button = tk.Button(side_panel, text="Удалить ребро", command=self.delete_edge_dialog)
        self.delete_edge_button.pack(pady=5)

        # Кнопка «Сгенерировать граф»
        self.generate_button = tk.Button(side_panel, text="Сгенерировать граф", command=self.generate_graph_dialog)
        self.generate_button.pack(pady=5)

        # Кнопка показать таблицу
        self.show_table_button = tk.Button(side_panel, text="Показать таблицу", command=self.toggle_table)
        self.show_table_button.pack(pady=5)

        # Кнопка «Стереть всё»
        self.clear_all_button = tk.Button(side_panel, text="Стереть всё", command=self.clear_all)
        self.clear_all_button.pack(pady=5)

        # ---------- Нижний Canvas: отображение результата TSP ----------
        self.canvas_bottom = tk.Canvas(main_frame, width=600, height=300, bg="white")
        self.canvas_bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---------- Таблица графа в правом нижнем углу ----------
        self.table_label = tk.Label(self.root, text="", font=("Courier", 10), justify="left", bg="lightgrey")
        self.table_visible = False
        self.table_label.place(relx=1.0, rely=1.0, anchor="se")

        # ---------- Данные о графе ----------
        self.vertex_positions = []  # список (x, y) для каждой вершины
        self.vertex_objects = []    # (oval_id, text_id) на верхнем Canvas
        self.adj_matrix = []        # матрица смежности (веса), None если ребра нет

        # Временное хранение при добавлении рёбер
        self.edge_first_vertex = None  # индекс первой выбранной вершины (или None)

    # ---------------------------------------------------------------------
    #                    ОБРАБОТКА КЛИКОВ И ДОБАВЛЕНИЯ
    # ---------------------------------------------------------------------
    def on_canvas_top_click(self, event):
        """Обработчик клика по верхнему canvas."""
        if self.mode.get() == "vertex":
            self.add_vertex(event.x, event.y)
        else:
            clicked_vertex = self.get_vertex_at_position(event.x, event.y)
            if clicked_vertex is not None:
                self.on_edge_vertex_clicked(clicked_vertex)

    def add_vertex(self, x, y):
        """Добавить вершину (x, y) и обновить матрицу смежности."""
        vertex_id = len(self.vertex_positions) + 1  # нумерация с 1
        self.vertex_positions.append((x, y))
        for row in self.adj_matrix:
            row.append(None)
        self.adj_matrix.append([None] * len(self.vertex_positions))
        self.draw_graph_on_top_canvas()

    def get_vertex_at_position(self, x, y, radius=10):
        """Вернуть индекс вершины, если (x,y) попало в её круг."""
        for i, (vx, vy) in enumerate(self.vertex_positions):
            if (vx - x)**2 + (vy - y)**2 <= radius**2:
                return i
        return None

    def on_edge_vertex_clicked(self, vertex_index):
        """Логика при добавлении рёбер: первая/вторая вершина."""
        if self.edge_first_vertex is None:
            self.edge_first_vertex = vertex_index
            self.highlight_vertex(vertex_index, True)
        else:
            if vertex_index == self.edge_first_vertex:
                messagebox.showwarning("Предупреждение", "Нельзя выбирать одну и ту же вершину.")
                self.highlight_vertex(self.edge_first_vertex, False)
                self.edge_first_vertex = None
                return

            direction = self.direction_var.get()
            w_str = self.weight_entry.get()
            try:
                w = float(w_str)
            except ValueError:
                messagebox.showerror("Ошибка", "Некорректный вес.")
                self.highlight_vertex(self.edge_first_vertex, False)
                self.edge_first_vertex = None
                return

            i = self.edge_first_vertex
            j = vertex_index

            if direction == "forward":
                self.adj_matrix[i][j] = w
            elif direction == "backward":
                self.adj_matrix[j][i] = w
            else:
                w1_str = simpledialog.askstring("Вес (i->j)",
                    f"Вес для направления {i+1}→{j+1} (по умолчанию {w_str}):")
                if w1_str is None or w1_str.strip() == "":
                    w1 = w
                else:
                    try:
                        w1 = float(w1_str)
                    except ValueError:
                        w1 = w

                w2_str = simpledialog.askstring("Вес (j->i)",
                    f"Вес для направления {j+1}→{i+1} (по умолчанию {w_str}):")
                if w2_str is None or w2_str.strip() == "":
                    w2 = w
                else:
                    try:
                        w2 = float(w2_str)
                    except ValueError:
                        w2 = w

                self.adj_matrix[i][j] = w1
                self.adj_matrix[j][i] = w2

            self.highlight_vertex(self.edge_first_vertex, False)
            self.edge_first_vertex = None
            self.draw_graph_on_top_canvas()

    def highlight_vertex(self, vertex_index, highlight=True):
        """Подсветить/убрать подсветку вершины (в режиме добавления рёбер)."""
        if 0 <= vertex_index < len(self.vertex_objects):
            oval_id, _ = self.vertex_objects[vertex_index]
            color = "yellow" if highlight else "lightblue"
            self.canvas_top.itemconfig(oval_id, fill=color)

    # ---------------------------------------------------------------------
    #                        УДАЛЕНИЕ ВЕРШИН/РЁБЕР
    # ---------------------------------------------------------------------
    def delete_vertex_dialog(self):
        """Диалог для удаления вершины."""
        if not self.vertex_positions:
            messagebox.showinfo("Удаление вершины", "В графе нет вершин.")
            return
        v_str = simpledialog.askstring("Удалить вершину", "Введите номер вершины (1..n):")
        if v_str is None:
            return
        try:
            v = int(v_str) - 1
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректный номер вершины.")
            return
        if v < 0 or v >= len(self.vertex_positions):
            messagebox.showerror("Ошибка", "Нет вершины с таким номером.")
            return
        self.delete_vertex(v)

    def delete_vertex(self, v_index):
        """Удалить вершину с индексом v_index (0-based)."""
        del self.adj_matrix[v_index]
        for row in self.adj_matrix:
            del row[v_index]
        del self.vertex_positions[v_index]
        del self.vertex_objects[v_index]
        self.draw_graph_on_top_canvas()

    def delete_edge_dialog(self):
        """Диалог для удаления ребра."""
        if not self.vertex_positions:
            messagebox.showinfo("Удаление ребра", "В графе нет вершин.")
            return
        v1_str = simpledialog.askstring("Удалить ребро", "Введите вершину-источник (1..n):")
        if v1_str is None:
            return
        v2_str = simpledialog.askstring("Удалить ребро", "Введите вершину-назначение (1..n):")
        if v2_str is None:
            return
        try:
            v1 = int(v1_str) - 1
            v2 = int(v2_str) - 1
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные номера вершин.")
            return
        if (v1 < 0 or v1 >= len(self.vertex_positions) or
            v2 < 0 or v2 >= len(self.vertex_positions)):
            messagebox.showerror("Ошибка", "Нет вершины с таким номером.")
            return
        self.delete_edge(v1, v2)
        self.vertex_objects.clear()
        self.draw_graph_on_top_canvas()

    def delete_edge(self, v1, v2):
        """Удалить ребро v1->v2."""
        self.adj_matrix[v1][v2] = None
        self.draw_graph_on_top_canvas()

    # ---------------------------------------------------------------------
    #                  СЛУЧАЙНАЯ ГЕНЕРАЦИЯ ГРАФА
    # ---------------------------------------------------------------------
    def generate_graph_dialog(self):
        """Диалог для генерации случайного графа."""
        n_str = simpledialog.askstring("Генерация графа", "Сколько вершин создать?")
        if n_str is None:
            return
        try:
            n = int(n_str)
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное число вершин.")
            return
        if n < 1:
            messagebox.showerror("Ошибка", "Число вершин должно быть >= 1.")
            return

        self.vertex_positions = []
        self.vertex_objects = []
        self.adj_matrix = []

        width = 950
        height = 350
        p = 1
        w_min, w_max = 1, 10

        for i in range(n):
            x = random.randint(50, width - 50)
            y = random.randint(50, height - 50)
            self.vertex_positions.append((x, y))

        self.adj_matrix = [[None]*n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j and random.random() < p:
                    w = random.randint(w_min, w_max)
                    self.adj_matrix[i][j] = float(w)

        self.draw_graph_on_top_canvas()

    def generate_graph(self, num_vertices):
        """Генерирует случайный полный граф с заданным количеством вершин"""
        self.clear_all()  # Очищаем текущий граф
    
        width = 950
        height = 350
    
        # Генерируем случайные позиции вершин
        for _ in range(num_vertices):
            x = random.randint(50, width - 50)
            y = random.randint(50, height - 50)
            self.add_vertex(x, y)
        
        # Создаем полную матрицу смежности со случайными весами
        n = len(self.vertex_positions)
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.adj_matrix[i][j] = float(random.randint(1, 100))
        
        self.draw_graph_on_top_canvas()

    # ---------------------------------------------------------------------
    #                     КНОПКА «СТЕРЕТЬ ВСЁ»
    # ---------------------------------------------------------------------
    def clear_all(self):
        """Стереть все данные о графе и очистить оба canvas."""
        self.vertex_positions.clear()
        self.vertex_objects.clear()
        self.adj_matrix.clear()
        self.canvas_top.delete("all")
        self.canvas_bottom.delete("all")
        self.table_label.config(text="")  # очищаем таблицу

    # ---------------------------------------------------------------------
    #                         ОТРИСОВКА ГРАФА
    # ---------------------------------------------------------------------
    def draw_graph_on_top_canvas(self):
        """Полностью перерисовать верхний canvas по данным self.adj_matrix."""
        self.canvas_top.delete("all")
        self.vertex_objects = []
        r = 8
        for i, (x, y) in enumerate(self.vertex_positions):
            oval_id = self.canvas_top.create_oval(x - r, y - r, x + r, y + r,
                                                  fill="lightblue", outline="black", width=2)
            text_id = self.canvas_top.create_text(x, y, text=str(i+1))
            self.vertex_objects.append((oval_id, text_id))
        n = len(self.vertex_positions)
        for i in range(n):
            for j in range(i+1, n):
                w1 = self.adj_matrix[i][j]
                w2 = self.adj_matrix[j][i]
                if w1 is not None and w2 is not None:
                    self.draw_directed_edge(i, j, w1, above=True)
                    self.draw_directed_edge(j, i, w2, above=False)
                elif w1 is not None:
                    self.draw_directed_edge(i, j, w1, above=True)
                elif w2 is not None:
                    self.draw_directed_edge(j, i, w2, above=True)
        # Обновляем таблицу графа
        self.update_table()

    def draw_directed_edge(self, i, j, weight, above=True):
        x1, y1 = self.vertex_positions[i]
        x2, y2 = self.vertex_positions[j]
        self.canvas_top.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=2)
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist == 0:
            return
        px = -dy
        py = dx
        mag = math.hypot(px, py)
        if mag == 0:
            return
        px /= mag
        py /= mag
        offset = 10 if above else -10
        lx = mx + px * offset
        ly = my + py * offset
        self.canvas_top.create_text(lx, ly, text=str(weight), fill="red")

    # ---------------------------------------------------------------------
    #                            ЗАПУСК TSP
    # ---------------------------------------------------------------------
    def run_tsp(self):
        n = len(self.vertex_positions)
        if n < 2:
            messagebox.showinfo("Результат", "Слишком мало вершин для TSP.")
            return
        best_path = None
        best_cost = float("inf")
        start_time = time.time()
        for start in range(n):
            path, cost = self.nearest_neighbor_tsp(start)
            if path is not None and cost < best_cost:
                best_cost = cost
                best_path = path
        total_time = time.time() - start_time
        if best_path is None:
            messagebox.showinfo("Результат", "Не удалось найти гамильтонов цикл.")
        else:
            self.draw_path_in_bottom_canvas(best_path, best_cost)
            print(f"Алгоритм: Ближайший сосед      | Вершин: {n} | Длина цикла: {best_cost:.2f} | Время: {total_time:.4f} сек")
  
    def nearest_neighbor_tsp(self, start):
        n = len(self.adj_matrix)
        visited = [False] * n
        visited[start] = True
        path = [start]
        current = start
        total_cost = 0
        for _ in range(n - 1):
            next_vertex = None
            min_dist = float("inf")
            for j in range(n):
                if not visited[j] and self.adj_matrix[current][j] is not None:
                    dist = self.adj_matrix[current][j]
                    if dist < min_dist:
                        min_dist = dist
                        next_vertex = j
            if next_vertex is None:
                return None, float("inf")
            visited[next_vertex] = True
            path.append(next_vertex)
            total_cost += self.adj_matrix[current][next_vertex]
            current = next_vertex
        if self.adj_matrix[current][start] is None:
            return None, float("inf")
        else:
            total_cost += self.adj_matrix[current][start]
            path.append(start)
            return path, total_cost

    # Модифицированный алгоритм TSP: на каждом шаге выбирается случайный кандидат,
    # чей вес не превышает минимальный на 10% (epsilon = 0.1)
    def run_tsp_modified(self):
        n = len(self.vertex_positions)
        if n < 2:
            messagebox.showinfo("Результат", "Слишком мало вершин для TSP.")
            return
        best_path = None
        best_cost = float("inf")
        start_time = time.time()
        for start in range(n):
            path, cost = self.nearest_neighbor_tsp_modified(start, epsilon=0.1)
            if path is not None and cost < best_cost:
                best_cost = cost
                best_path = path
        total_time = time.time() - start_time
        if best_path is None:
            messagebox.showinfo("Результат", "Не удалось найти гамильтонов цикл.")
        else:
            self.draw_path_in_bottom_canvas(best_path, best_cost)
            print(f"Алгоритм: Мод. ближайший сосед | Вершин: {n} | Длина цикла: {best_cost:.2f} | Время: {total_time:.4f} сек")

    def nearest_neighbor_tsp_modified(self, start, epsilon=0.1):
        n = len(self.adj_matrix)
        visited = [False] * n
        visited[start] = True
        path = [start]
        current = start
        total_cost = 0
        for _ in range(n - 1):
            candidates = []
            min_dist = float("inf")
            for j in range(n):
                if not visited[j] and self.adj_matrix[current][j] is not None:
                    dist = self.adj_matrix[current][j]
                    if dist < min_dist:
                        min_dist = dist
                        candidates = [j]
                    elif dist <= min_dist * (1 + epsilon):
                        candidates.append(j)
            if not candidates:
                return None, float("inf")
            next_vertex = random.choice(candidates)
            visited[next_vertex] = True
            path.append(next_vertex)
            total_cost += self.adj_matrix[current][next_vertex]
            current = next_vertex
        if self.adj_matrix[current][start] is None:
            return None, float("inf")
        else:
            total_cost += self.adj_matrix[current][start]
            path.append(start)
            return path, total_cost

    def draw_path_in_bottom_canvas(self, path, cost):
        self.canvas_bottom.delete("all")
        r = 8
        for i, (x, y) in enumerate(self.vertex_positions):
            self.canvas_bottom.create_oval(x - r, y - r, x + r, y + r,
                                           fill="pink", outline="black", width=2)
            self.canvas_bottom.create_text(x, y, text=str(i+1))
        for k in range(len(path) - 1):
            i = path[k]
            j = path[k+1]
            x1, y1 = self.vertex_positions[i]
            x2, y2 = self.vertex_positions[j]
            self.canvas_bottom.create_line(x1, y1, x2, y2,
                                           arrow=tk.LAST, fill="red", width=2)
        self.canvas_bottom.create_text(100, 15, text=f"Вес цикла: {cost:.2f}",
                                       fill="blue", font=("Arial", 14, "bold"))
        
    def toggle_table(self):
        if self.table_visible:
            self.table_label.place_forget()
            self.table_visible = False
        else:
            self.update_table()
            self.table_label.place(relx=1.0, rely=1.0, anchor="se")
            self.table_visible = True
        
    def run_simulated_annealing(self):
        def total_distance(path):
            cost = 0
            for i in range(len(path) - 1):
                w = self.adj_matrix[path[i]][path[i + 1]]
                if w is None:
                    return float("inf")
                cost += w
            last_leg = self.adj_matrix[path[-1]][path[0]]
            if last_leg is None:
                return float("inf")
            cost += last_leg
            return cost

        def random_neighbor(path):
            a, b = random.sample(range(len(path)), 2)
            new_path = path[:]
            new_path[a], new_path[b] = new_path[b], new_path[a]
            return new_path

        n = len(self.vertex_positions)
        if n < 2:
            messagebox.showinfo("Результат", "Слишком мало вершин.")
            return

        for i in range(n):
            for j in range(n):
                if i != j and self.adj_matrix[i][j] is None:
                    messagebox.showwarning("Ошибка", "Граф неполный! Добавьте рёбра между всеми вершинами.")
                    return

        start_time = time.time()
        current_path = list(range(n))
        random.shuffle(current_path)
        current_cost = total_distance(current_path)

        if current_cost == float("inf"):
            messagebox.showinfo("Результат", "Не удалось найти гамильтонов цикл.")
            return

        T = 1000
        T_min = 1e-4
        alpha = 0.995

        while T > T_min:
            for _ in range(100):
                new_path = random_neighbor(current_path)
                new_cost = total_distance(new_path)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_path = new_path
                    current_cost = new_cost
            T *= alpha

        current_path.append(current_path[0])
        self.draw_path_in_bottom_canvas(current_path, current_cost)
        total_time = time.time() - start_time
        print(f"Алгоритм: Имитация отжига      | Вершин: {n} | Длина цикла: {current_cost:.2f} | Время: {total_time:.4f} сек")
                          
        
    def run_fast_simulated_annealing(self):
        def total_distance(path):
            cost = 0
            for i in range(len(path) - 1):
                w = self.adj_matrix[path[i]][path[i + 1]]
                if w is None:
                    return float("inf")
                cost += w
            last_leg = self.adj_matrix[path[-1]][path[0]]
            if last_leg is None:
                return float("inf")
            cost += last_leg
            return cost

        def random_neighbor(path):
            a, b = random.sample(range(len(path)), 2)
            new_path = path[:]
            new_path[a], new_path[b] = new_path[b], new_path[a]
            return new_path

        n = len(self.vertex_positions)
        if n < 2:
            messagebox.showinfo("Результат", "Слишком мало вершин.")
            return

        for i in range(n):
            for j in range(n):
                if i != j and self.adj_matrix[i][j] is None:
                    messagebox.showwarning("Ошибка", "Граф неполный! Добавьте рёбра между всеми вершинами.")
                    return

        start_time = time.time()
        current_path = list(range(n))
        random.shuffle(current_path)
        current_cost = total_distance(current_path)

        if current_cost == float("inf"):
            messagebox.showinfo("Результат", "Не удалось найти гамильтонов цикл.")
            return

        T0 = 1000
        max_iter = 5000

        for iteration in range(1, max_iter):
            T = T0 / iteration
            new_path = random_neighbor(current_path)
            new_cost = total_distance(new_path)
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_path = new_path
                current_cost = new_cost

        current_path.append(current_path[0])
        self.draw_path_in_bottom_canvas(current_path, current_cost)
        total_time = time.time() - start_time
        print(f"Алгоритм: Сверхбыстрый отжиг   | Вершин: {n} | Длина цикла: {current_cost:.2f} | Время: {total_time:.4f} сек")

    # ---------------------- Муравьиный алгоритм ----------------------
    # --- Муравьиный алгоритм ---
    def run_ant_colony(self):
        n = len(self.vertex_positions)
        if n < 2:
            messagebox.showinfo("Результат", "Слишком мало вершин для TSP.")
            return
        for i in range(n):
            for j in range(n):
                if i != j and self.adj_matrix[i][j] is None:
                    messagebox.showwarning("Ошибка", "Граф неполный! Добавьте рёбра.")
                    return
        start_time = time.time()
        best_path, best_cost = self.ant_colony_tsp(num_ants=n, num_iters=100, alpha=1.0, beta=2.0, evap=0.5, Q=100.0)
        total_time = time.time() - start_time
        self.draw_path_in_bottom_canvas(best_path, best_cost)
        print(f"Алгоритм: Муравьиный алгоритм  | Вершин: {n} | Длина цикла: {best_cost:.2f} | Время: {total_time:.4f} сек") 

    def ant_colony_tsp(self,num_ants,num_iters,alpha,beta,evap,Q):
        n=len(self.adj_matrix)
        tau=[[1.0]*n for _ in range(n)]
        dist=self.adj_matrix
        best_path=None;best_cost=math.inf
        for _ in range(num_iters):
            all_paths=[];all_costs=[]
            for _ in range(num_ants):
                start=random.randrange(n);visited={start};path=[start];curr=start;cost=0.0
                while len(path)<n:
                    probs=[];den=0.0
                    for j in range(n):
                        if j not in visited:
                            t=tau[curr][j]**alpha;e=(1.0/dist[curr][j])**beta
                            den+=t*e;probs.append((j,t*e))
                    if den==0:break
                    r=random.random();cum=0.0;next_v=None
                    for j,v in probs:
                        cum+=v/den
                        if r<=cum:next_v=j;break
                    if next_v is None:next_v=probs[-1][0]
                    path.append(next_v);visited.add(next_v);cost+=dist[curr][next_v];curr=next_v
                cost+=dist[curr][path[0]];path.append(path[0])
                all_paths.append(path);all_costs.append(cost)
                if cost<best_cost:best_cost=cost;best_path=path
            # испарение
            for i in range(n):
                for j in range(n):tau[i][j]*=(1-evap)
            # депонирование
            for path,cost in zip(all_paths,all_costs):
                dQ=Q/cost
                for k in range(len(path)-1):
                    i,j=path[k],path[k+1];tau[i][j]+=dQ
        return best_path,best_cost

    # --- Элитные муравьи ---
    def run_elite_ant_colony(self):
        n = len(self.vertex_positions)
        if n < 2:
            messagebox.showinfo("Результат", "Слишком мало вершин для TSP.")
            return
        for i in range(n):
            for j in range(n):
                if i != j and self.adj_matrix[i][j] is None:
                    messagebox.showwarning("Ошибка", "Граф неполный! Добавьте рёбра.")
                    return
        start_time = time.time()
        best_path, best_cost = self.elite_ant_colony_tsp(num_ants=n, num_iters=100, alpha=1.0, beta=2.0, evap=0.5, Q=100.0, elitist=5)
        total_time = time.time() - start_time
        self.draw_path_in_bottom_canvas(best_path, best_cost)
        print(f"Алгоритм: Элитные муравьи      | Вершин: {n} | Длина цикла: {best_cost:.2f} | Время: {total_time:.4f} сек") 

    def elite_ant_colony_tsp(self,num_ants,num_iters,alpha,beta,evap,Q,elitist):
        n=len(self.adj_matrix)
        tau=[[1.0]*n for _ in range(n)]
        dist=self.adj_matrix
        best_path=None;best_cost=math.inf
        for _ in range(num_iters):
            all_paths=[];all_costs=[]
            for _ in range(num_ants):
                start=random.randrange(n);visited={start};path=[start];curr=start;cost=0.0
                while len(path)<n:
                    probs=[];den=0.0
                    for j in range(n):
                        if j not in visited:
                            t=tau[curr][j]**alpha;e=(1.0/dist[curr][j])**beta
                            den+=t*e;probs.append((j,t*e))
                    if den==0:break
                    r=random.random();cum=0.0;next_v=None
                    for j,v in probs:
                        cum+=v/den
                        if r<=cum:next_v=j;break
                    if next_v is None:next_v=probs[-1][0]
                    path.append(next_v);visited.add(next_v);cost+=dist[curr][next_v];curr=next_v
                cost+=dist[curr][path[0]];path.append(path[0])
                all_paths.append(path);all_costs.append(cost)
                if cost<best_cost:best_cost=cost;best_path=path
            # испарение
            for i in range(n):
                for j in range(n):tau[i][j]*=(1-evap)
            # депонирование обычное
            for path,cost in zip(all_paths,all_costs):
                dQ=Q/cost
                for k in range(len(path)-1):
                    i,j=path[k],path[k+1];tau[i][j]+=dQ
            # депонирование элиты
            if best_path:
                elit_deposit=elitist*(Q/best_cost)
                for k in range(len(best_path)-1):
                    i,j=best_path[k],best_path[k+1];tau[i][j]+=elit_deposit
        return best_path,best_cost

    def update_table(self):
        """Обновить таблицу графа (матрицу смежности) в виджете table_label."""
        table_str = "Adjacency Matrix:\n"
        n = len(self.adj_matrix)
        header = "     " + " ".join(f"{j+1:>5}" for j in range(n)) + "\n"
        table_str += header
        for i in range(n):
            row_str = f"{i+1:>3} " + " ".join(f"{self.adj_matrix[i][j]:5.1f}" if self.adj_matrix[i][j] is not None else "  -  " for j in range(n)) + "\n"
            table_str += row_str
        self.table_label.config(text=table_str)

def test():
    v = [10, 30, 50, 100]
    for i in v:
        temp_root = tk.Tk()
        temp_root.withdraw()  
        app = TSPApp(temp_root)
        app.generate_graph(i)

        algorithms = [
            ("Ближайший сосед", app.run_tsp),
            ("Мод. ближайший сосед", app.run_tsp_modified),
            ("Имитация отжига", app.run_simulated_annealing),
            ("Сверхбыстрый отжиг", app.run_fast_simulated_annealing),
            ("Муравьиный алгоритм", app.run_ant_colony),
            ("Элитные муравьи", app.run_elite_ant_colony)
        ]
        
        for name, algorithm in algorithms:
            algorithm()

        print()
            
        temp_root.destroy()

def main():
    test()
    #root = tk.Tk()
    #app = TSPApp(root)
    #root.mainloop()

if __name__ == "__main__":
    main()