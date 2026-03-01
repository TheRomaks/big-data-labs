def test_hypotheses(df, hypotheses_list, target_column='Outcome'):
    results = []

    for name, column, threshold in hypotheses_list:
        print(f"\nГипотеза: {name} влияет на целевую переменную")

        if column not in df.columns:
            print(f"Предупреждение: колонка {column} не найдена")
            continue

        if target_column not in df.columns:
            print(f"Ошибка: целевая колонка {target_column} не найдена")
            continue

        corr = df[column].corr(df[target_column])
        print(f"Корреляция {column} и {target_column}: {corr:.4f}")

        if abs(corr) > threshold:
            print("Гипотеза подтверждена")
            confirmed = True
        else:
            print("Гипотеза не подтверждена")
            confirmed = False

        results.append({
            'name': name,
            'column': column,
            'correlation': corr,
            'threshold': threshold,
            'confirmed': confirmed
        })

    return results

def print_hypothesis_summary(results):
    for res in results:
        status = "Подтверждена" if res['confirmed'] else "Не подтверждена"
        print(f"{res['name']}: {status} (corr={res['correlation']:.4f})")

    confirmed_count = sum(1 for r in results if r['confirmed'])
    print(f"\nВсего подтверждено: {confirmed_count}/{len(results)}")

    return results