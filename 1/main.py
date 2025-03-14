import tkinter as tk
from tkinter import messagebox, simpledialog
import random
import math

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
        tk.Label(side_panel, text="Направление:").pack(anchor=tk.W)
        self.direction_var = tk.StringVar(value="forward")  
        tk.Radiobutton(side_panel, text="1 → 2",
                       variable=self.direction_var, value="forward").pack(anchor=tk.W)
        tk.Radiobutton(side_panel, text="2 → 1",
                       variable=self.direction_var, value="backward").pack(anchor=tk.W)
        tk.Radiobutton(side_panel, text="Обе (разные веса)",
                       variable=self.direction_var, value="both").pack(anchor=tk.W)

        # Поле для ввода веса (основного)
        tk.Label(side_panel, text="Вес (для одного направления):").pack(anchor=tk.W)
        self.weight_entry = tk.Entry(side_panel, width=5)
        self.weight_entry.pack(anchor=tk.W)
        self.weight_entry.insert(0, "1")  # по умолчанию 1

        # Кнопка «Запустить TSP»
        self.tsp_button = tk.Button(side_panel, text="Запустить TSP", command=self.run_tsp)
        self.tsp_button.pack(pady=5)

        # Кнопка «Удалить вершину»
        self.delete_vertex_button = tk.Button(side_panel, text="Удалить вершину", command=self.delete_vertex_dialog)
        self.delete_vertex_button.pack(pady=5)

        # Кнопка «Удалить ребро»
        self.delete_edge_button = tk.Button(side_panel, text="Удалить ребро", command=self.delete_edge_dialog)
        self.delete_edge_button.pack(pady=5)

        # Кнопка «Сгенерировать граф»
        self.generate_button = tk.Button(side_panel, text="Сгенерировать граф", command=self.generate_graph_dialog)
        self.generate_button.pack(pady=5)

        # Кнопка «Стереть всё»
        self.clear_all_button = tk.Button(side_panel, text="Стереть всё", command=self.clear_all)
        self.clear_all_button.pack(pady=5)

        # ---------- Нижний Canvas: отображение результата TSP ----------
        self.canvas_bottom = tk.Canvas(main_frame, width=600, height=300, bg="white")
        self.canvas_bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---------- Данные о графе ----------
        self.vertex_positions = []  # список (x, y) для каждой вершины
        self.vertex_objects = []    # (oval_id, text_id) на верхнем Canvas
        self.adj_matrix = []        # матрица смежности (веса). None если ребра нет

        # Временное хранение при добавлении рёбер
        self.edge_first_vertex = None  # индекс первой выбранной вершины (или None)

    # ---------------------------------------------------------------------
    #                    ОБРАБОТКА КЛИКОВ И ДОБАВЛЕНИЯ
    # ---------------------------------------------------------------------
    def on_canvas_top_click(self, event):
        """Обработчик клика по верхнему canvas."""
        if self.mode.get() == "vertex":
            # Добавляем новую вершину
            self.add_vertex(event.x, event.y)
        else:
            # Режим добавления рёбер
            clicked_vertex = self.get_vertex_at_position(event.x, event.y)
            if clicked_vertex is not None:
                self.on_edge_vertex_clicked(clicked_vertex)

    def add_vertex(self, x, y):
        """Добавить вершину (x, y) и обновить матрицу смежности."""
        vertex_id = len(self.vertex_positions) + 1  # нумерация с 1
        self.vertex_positions.append((x, y))

        # Расширим матрицу смежности
        for row in self.adj_matrix:
            row.append(None)
        self.adj_matrix.append([None] * vertex_id)

        # Перерисуем граф
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
            # Выбрали первую вершину
            self.edge_first_vertex = vertex_index
            self.highlight_vertex(vertex_index, True)
        else:
            # Выбрали вторую вершину
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
                # i->j
                self.adj_matrix[i][j] = w
            elif direction == "backward":
                # j->i
                self.adj_matrix[j][i] = w
            else:
                # "both": запросить отдельно вес для i->j и j->i
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

            # Перерисуем граф
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
        v_str = simpledialog.askstring("Удалить вершину", 
                                       "Введите номер вершины (1..n):")
        if v_str is None:
            return
        try:
            v = int(v_str) - 1  # перевод в 0-based
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректный номер вершины.")
            return
        if v < 0 or v >= len(self.vertex_positions):
            messagebox.showerror("Ошибка", "Нет вершины с таким номером.")
            return

        self.delete_vertex(v)

    def delete_vertex(self, v_index):
        """Удалить вершину с индексом v_index (0-based)."""
        # Удаляем из матрицы смежности соответствующую строку и столбец
        del self.adj_matrix[v_index]
        for row in self.adj_matrix:
            del row[v_index]

        # Удаляем из списков
        del self.vertex_positions[v_index]
        del self.vertex_objects[v_index]

        # Полностью перерисуем
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

    def delete_edge(self, v1, v2):
        """Удалить ребро v1->v2."""
        self.adj_matrix[v1][v2] = None
        self.draw_graph_on_top_canvas()

    # ---------------------------------------------------------------------
    #                  СЛУЧАЙНАЯ ГЕНЕРАЦИЯ ГРАФА
    # ---------------------------------------------------------------------
    def generate_graph_dialog(self):
        """Диалог для генерации случайного графа."""
        n_str = simpledialog.askstring("Генерация графа", 
                                       "Сколько вершин создать?")
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

        # Удаляем текущий граф
        self.vertex_positions = []
        self.vertex_objects = []
        self.adj_matrix = []

        # Параметры для генерации
        width = 1000
        height = 350
        # вероятность существования ребра (i->j)
        p = 0.4  
        # диапазон весов
        w_min, w_max = 1, 10

        # Создадим n вершин со случайными координатами
        for i in range(n):
            x = random.randint(50, width - 50)
            y = random.randint(50, height - 50)
            self.vertex_positions.append((x, y))

        # Создадим матрицу смежности n x n
        self.adj_matrix = [[None]*n for _ in range(n)]

        # Генерируем рёбра
        for i in range(n):
            for j in range(n):
                if i != j:
                    # С вероятностью p делаем i->j
                    if random.random() < p:
                        w = random.randint(w_min, w_max)
                        self.adj_matrix[i][j] = float(w)

        # Перерисуем
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

    # ---------------------------------------------------------------------
    #                         ОТРИСОВКА ГРАФА
    # ---------------------------------------------------------------------
    def draw_graph_on_top_canvas(self):
        """Полностью перерисовать верхний canvas по данным self.adj_matrix."""
        self.canvas_top.delete("all")
        self.vertex_objects = []

        # Сначала рисуем вершины
        r = 8
        for i, (x, y) in enumerate(self.vertex_positions):
            oval_id = self.canvas_top.create_oval(x - r, y - r, x + r, y + r,
                                                  fill="lightblue", outline="black", width=2)
            text_id = self.canvas_top.create_text(x, y, text=str(i+1))
            self.vertex_objects.append((oval_id, text_id))

        # Затем рёбра
        n = len(self.vertex_positions)
        # Чтобы корректно отрисовать «туда и обратно», пройдём только i<j,
        # и для пары (i,j) посмотрим adj[i][j] и adj[j][i].
        for i in range(n):
            for j in range(i+1, n):
                w1 = self.adj_matrix[i][j]
                w2 = self.adj_matrix[j][i]
                # Если оба направления есть
                if w1 is not None and w2 is not None:
                    # i->j (с весом над линией)
                    self.draw_directed_edge(i, j, w1, above=True)
                    # j->i (с весом под линией)
                    self.draw_directed_edge(j, i, w2, above=False)
                elif w1 is not None:
                    # только i->j
                    self.draw_directed_edge(i, j, w1, above=True)
                elif w2 is not None:
                    # только j->i
                    self.draw_directed_edge(j, i, w2, above=True)

    def draw_directed_edge(self, i, j, weight, above=True):
        """
        Нарисовать стрелку i->j на canvas_top, 
        сместив подпись веса "над" (above=True) или "под" (above=False) линией.
        """
        x1, y1 = self.vertex_positions[i]
        x2, y2 = self.vertex_positions[j]

        # Рисуем линию со стрелкой
        self.canvas_top.create_line(x1, y1, x2, y2,
                                    arrow=tk.LAST, width=2)

        # Найдём середину
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        # Вектор i->j
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist == 0:
            return

        # Перпендикуляр
        px = -dy
        py = dx
        mag = math.hypot(px, py)
        if mag == 0:
            return
        px /= mag
        py /= mag

        # Смещение для текста
        offset = 10
        if not above:
            offset = -10

        lx = mx + px * offset
        ly = my + py * offset

        self.canvas_top.create_text(lx, ly, text=str(weight), fill="red")

    # ---------------------------------------------------------------------
    #                            ЗАПУСК TSP
    # ---------------------------------------------------------------------
    def run_tsp(self):
        """Запустить алгоритм ближайшего соседа по всем вершинам, найти лучший цикл."""
        n = len(self.vertex_positions)
        if n < 2:
            messagebox.showinfo("Результат", "Слишком мало вершин для TSP.")
            return

        best_path = None
        best_cost = float("inf")

        for start in range(n):
            path, cost = self.nearest_neighbor_tsp(start)
            if path is not None and cost < best_cost:
                best_cost = cost
                best_path = path

        if best_path is None:
            messagebox.showinfo("Результат", "Не удалось найти гамильтонов цикл.")
        else:
            self.draw_path_in_bottom_canvas(best_path, best_cost)

    def nearest_neighbor_tsp(self, start):
        """Метод ближайшего соседа, начиная с вершины start. Возвращает (путь, вес) или (None, inf)."""
        n = len(self.adj_matrix)
        visited = [False]*n
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
            total_cost += min_dist
            current = next_vertex

        # замыкаем цикл?
        if self.adj_matrix[current][start] is None:
            return None, float("inf")
        else:
            total_cost += self.adj_matrix[current][start]
            path.append(start)
            return path, total_cost

    def draw_path_in_bottom_canvas(self, path, cost):
        """Отобразить найденный путь в нижнем canvas."""
        self.canvas_bottom.delete("all")

        # Рисуем вершины
        r = 8
        for i, (x, y) in enumerate(self.vertex_positions):
            self.canvas_bottom.create_oval(x - r, y - r, x + r, y + r,
                                           fill="pink", outline="black", width=2)
            self.canvas_bottom.create_text(x, y, text=str(i+1))

        # Рисуем рёбра по path
        for k in range(len(path) - 1):
            i = path[k]
            j = path[k+1]
            x1, y1 = self.vertex_positions[i]
            x2, y2 = self.vertex_positions[j]
            self.canvas_bottom.create_line(x1, y1, x2, y2,
                                           arrow=tk.LAST, fill="red", width=2)

        # Выводим вес цикла
        self.canvas_bottom.create_text(100, 15, text=f"Вес цикла: {cost:.2f}",
                                       fill="blue", font=("Arial", 14, "bold"))

def main():
    root = tk.Tk()
    app = TSPApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
